from devtools.debug import Debug, DebugArgument
from deepseek_tokenizer import ds_token
import tiktoken
from typing import Any, Optional


class TokenDebugArgument(DebugArgument):
    def __init__(self, value: Any, *, name: Optional[str] = None, **extra: Any) -> None:
        super().__init__(value, name=name, **extra)
        if isinstance(value, str):
            # encoding = tiktoken.get_encoding("cl100k_base")
            encoding = ds_token
            token_count = len(encoding.encode(value))
            self.extra.append(("tokens", token_count))


class TokenDebug(Debug):
    output_class = Debug.output_class
    output_class.arg_class = TokenDebugArgument


debug = TokenDebug()
