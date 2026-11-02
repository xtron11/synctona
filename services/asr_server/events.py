from enum import Enum, auto
from dataclasses import dataclass

class TranscriptType(Enum):
    PARTIAL = auto(); FINAL = auto(); ENDPOINT = auto()

@dataclass
class TranscriptEvent:
    type: TranscriptType; text: str; language: str = ""; is_sentence_end: bool = False