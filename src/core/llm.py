"""LLM integration module for intent compilation."""

import json
import urllib.request
import urllib.error
from typing import Optional, List, Dict
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Base class for LLM integrations."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text completion."""
        pass

    def is_available(self) -> bool:
        """Check if the LLM backend is available."""
        return True


class LlamaCppLLM(BaseLLM):
    """Local LLM using llama.cpp server (OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8082",
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeout: int = 120
    ):
        """
        Initialize llama.cpp server connection.

        Args:
            base_url: llama.cpp server URL (e.g., "http://localhost:8082")
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if llama.cpp server is running."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            # Try the /v1/models endpoint as fallback
            try:
                req = urllib.request.Request(f"{self.base_url}/v1/models")
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status == 200
            except Exception:
                return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using llama.cpp server.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]

        except urllib.error.URLError as e:
            raise RuntimeError(
                f"llama.cpp server not reachable at {self.base_url}. "
                f"Start it with: llama-server.exe --model <path> --port 8082\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")

    def get_model_info(self) -> Optional[Dict]:
        """Get info about the loaded model."""
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None


class OllamaLLM(BaseLLM):
    """Local LLM using Ollama."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize Ollama LLM.

        Args:
            model: Model name (e.g., "llama3.1:8b", "mistral:7b", "qwen3:8b")
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _load_client(self):
        """Lazy load Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama
            except ImportError:
                raise ImportError(
                    "ollama not installed. Install with: pip install ollama"
                )
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            self._load_client()
            return True
        except Exception:
            return False

    def ensure_model(self) -> bool:
        """Ensure the model is downloaded."""
        client = self._load_client()
        try:
            print(f"🤖 Ensuring model {self.model} is available...")
            client.pull(self.model)
            print(f"✅ Model {self.model} ready!")
            return True
        except Exception as e:
            print(f"⚠️ Could not pull model: {e}")
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        client = self._load_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")

    def list_models(self) -> List[str]:
        """List available local models."""
        client = self._load_client()
        try:
            models = client.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception:
            return []


def get_default_llm(
    prefer_llamacpp: bool = True,
    llamacpp_url: str = "http://localhost:8082",
    ollama_model: str = "llama3.1:8b"
) -> BaseLLM:
    """
    Get the best available LLM backend.

    Args:
        prefer_llamacpp: Try llama.cpp first if True
        llamacpp_url: URL for llama.cpp server
        ollama_model: Ollama model to use as fallback

    Returns:
        Available LLM instance
    """
    if prefer_llamacpp:
        llm = LlamaCppLLM(base_url=llamacpp_url)
        if llm.is_available():
            print(f"✅ Using llama.cpp server at {llamacpp_url}")
            return llm
        print(f"⚠️ llama.cpp not available at {llamacpp_url}, trying Ollama...")

    try:
        llm = OllamaLLM(model=ollama_model)
        if llm.is_available():
            print(f"✅ Using Ollama with model {ollama_model}")
            return llm
    except ImportError:
        pass

    # Default to llama.cpp even if not available (will error on use)
    print(f"⚠️ No LLM backend found. Please start llama.cpp server or install Ollama.")
    return LlamaCppLLM(base_url=llamacpp_url)


class IntentCompiler:
    """Compile multimodal signals into clear intent prompts."""

    SYSTEM_PROMPT = """Clean up the user's speech. Remove filler words. Keep their EXACT meaning.

RULES:
1. Keep what they said - just cleaner
2. Remove: um, uh, like, you know, okay, basically, so, whatever
3. If emotion is negative (angry/worried/sad) → add "please" or "urgently"
4. If emotion is positive (happy) → keep it casual
5. Output ONLY the cleaned text, nothing else
6. If the input is unclear, just clean it minimally - do NOT invent content"""

    def __init__(self, llm: Optional[BaseLLM] = None, llamacpp_url: str = "http://localhost:8082"):
        """
        Initialize intent compiler.

        Args:
            llm: LLM instance (defaults to auto-detected best option)
            llamacpp_url: URL for llama.cpp server if auto-detecting
        """
        self.llm = llm or get_default_llm(llamacpp_url=llamacpp_url)

    def compile(
        self,
        text: str,
        context: str = "",
        emotion: Optional[str] = None,
        speaking_style: Optional[str] = None,
        gesture: Optional[str] = None
    ) -> str:
        """
        Compile raw text into clear intent.

        Args:
            text: Raw transcribed text
            context: Additional context
            emotion: Detected emotion
            speaking_style: Description of speaking style
            gesture: Detected gesture

        Returns:
            Compiled intent prompt
        """
        # Build the prompt with clear examples format
        parts = [f'Input: "{text}"']

        if context:
            parts.append(f"(Context: {context})")

        if emotion:
            parts.append(f"(User emotion: {emotion})")

        if speaking_style:
            parts.append(f"(Speaking style: {speaking_style})")

        if gesture:
            parts.append(f"(Gesture: {gesture})")

        parts.append("Output:")

        prompt = "\n".join(parts)

        result = self.llm.generate(prompt, self.SYSTEM_PROMPT)

        # Clean up the result - remove any unwanted prefixes
        result = result.strip()
        for prefix in ["Output:", "Instruction:", "Result:", "**", "```"]:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        # Remove trailing markdown/formatting
        result = result.rstrip("`*")

        return result.strip()

