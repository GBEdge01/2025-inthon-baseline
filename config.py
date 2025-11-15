from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenizerConfig:
    """토크나이저 관련 설정"""
    input_chars: Optional[list] = None  # 입력 문자 집합 (None이면 기본값 사용)
    output_chars: Optional[list] = None  # 출력 문자 집합 (None이면 기본값 사용)
    add_special: bool = True  # 특수 토큰(PAD, BOS, EOS) 추가 여부
    max_input_length: int = 64  # 입력 시퀀스 고정 길이
    max_output_length: int = 8  # 출력 숫자 자릿수


@dataclass
class ModelConfig:
    """모델 아키텍처 관련 설정"""
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_rel_distance: int = 64
    output_length: int = 8  # 분류할 출력 자릿수


@dataclass
class TrainConfig:
    """학습 관련 설정"""
    max_train_steps: Optional[int] = None
    lr: float = 1e-3
    valid_every: int = 50
    max_gen_len: int = 32
    show_valid_samples: int = 5
    num_epochs: int = 4
    save_best_path: Optional[str] = None