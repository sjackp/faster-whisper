import os
from faster_whisper import WhisperModel
from typing import Optional, Tuple, List, Union

class WhisperTranscriber:
    def __init__(self, model_size: str = "large-v3", device: str = "auto", compute_type: str = "default"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model is None:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path: Union[str, bytes], language: Optional[str] = None, beam_size: int = 5, word_timestamps: bool = False, vad_filter: bool = True) -> Tuple[str, List[dict]]:
        """
        Transcribe an audio file and return the transcript and segment metadata.
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter
        )
        transcript = "".join(segment.text for segment in segments).strip()
        segment_data = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                    for w in (segment.words or [])
                ]
            }
            for segment in segments
        ]
        return transcript, segment_data 