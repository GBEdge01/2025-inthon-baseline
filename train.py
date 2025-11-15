from __future__ import annotations
from typing import List, Any, Tuple
from config import TrainConfig, ModelConfig, TokenizerConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from dataloader import (
    ArithmeticDataset,  # 사칙연산 데이터를 만들어주는 Dataset
    get_dataloader,     # Dataset을 받아서 DataLoader로 바꿔주는 함수
)
from do_not_edit.metric import compute_metrics  # EM, TES 같은 간단한 성능 지표

from model import (
    TinySeq2Seq,
    CharTokenizer,      # 문자 단위 토크나이저
    tokenize_batch,     # batch(dict)를 토크나이즈 + 패딩까지 해주는 함수
    digits_to_string,
    INPUT_CHARS,        # 입력 문자 집합
    OUTPUT_CHARS,       # 출력 문자 집합
)


# ======================================================================================
# 1. 학습 루프
# ======================================================================================
def train_loop(
    model: nn.Module,           # 학습할 모델
    dataloader: DataLoader,     # DataLoader를 직접 전달받아 사용합니다.
    input_tokenizer: CharTokenizer,      # (tokenize_batch 유틸리티를 위해 유지)
    output_tokenizer: CharTokenizer,     # 출력 문자 토크나이저
    device: torch.device,       # cpu 또는 cuda
    val_dataloader: DataLoader | None = None,  # 별도 검증 DataLoader (None이면 훈련 배치로 검증)
    *,
    train_config: TrainConfig,  # 학습 설정
    model_config: ModelConfig,  # 모델 설정
    tokenizer_config: TokenizerConfig,  # 토크나이저 설정
):
    # 모델을 GPU/CPU로 보냄
    model.to(device)

    # 옵티마이저: AdamW는 Adam + weight decay가 들어간 버전
    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr)

    # 각 자리수에 대한 CrossEntropy loss
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    model.train()  # 학습 모드로 전환 (Dropout 등 켜짐)

    # tqdm은 진행 상황을 예쁘게 보여주는 라이브러리입니다.
    pbar = tqdm(total=train_config.max_train_steps if train_config.max_train_steps is not None else None, desc="train", unit="step", ncols=120, dynamic_ncols=True, leave=True)

    # best EM 추적용 변수 (None이 아니면 개선 시 모델 저장)
    best_em = float("-inf")

    for epoch in range(train_config.num_epochs):
        # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.
        if train_config.max_train_steps is not None and step >= train_config.max_train_steps: break
        pbar.write(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")

        # 실제로 배치를 하나씩 뽑아서 학습하는 부분입니다.
        for batch in dataloader:
            # --------------------------------------------------------------
            # 1) 토크나이즈 & 텐서로 변환
            #    입력은 후위표기 토큰 ID, 출력은 고정 길이 숫자 시퀀스입니다.
            # --------------------------------------------------------------
            batch_tensors = tokenize_batch(
                batch,
                input_tokenizer,
                output_tokenizer,
                max_input_length=tokenizer_config.max_input_length,
                max_output_length=tokenizer_config.max_output_length,
            )
            src = batch_tensors.src.to(device)
            target_digits = batch_tensors.tgt.to(device)
            src_pad_mask = src.eq(input_tokenizer.pad_id)

            # Forward: Transformer Encoder 결과로 자리별 logits 생성
            logits = model(src, src_pad_mask)

            # --------------------------------------------------------------
            # 4) Loss 계산
            # --------------------------------------------------------------
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),  # (B*T, V)
                target_digits.view(-1),             # (B*T,)
            )

            # --------------------------------------------------------------
            # 5) Backward + optimizer step
            # --------------------------------------------------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient 폭주 방지를 위해 클리핑
            optim.step()
            optim.zero_grad()

            step += 1 

            # --------------------------------------------------------------
            # 6) Validation
            # --------------------------------------------------------------
            if step % train_config.valid_every == 0:
                model.eval()  # 평가 모드
                # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.
                with torch.no_grad():
                    preds_all: List[str] = []
                    targets_all: List[str] = []
                    inputs_all: List[str] = [] # 검증 데이터셋의 입력, 정답, 예측 결과를 저장할 리스트

                    for val_batch in val_dataloader: # 검증 데이터셋을 순회하며 각 배치에 대해 검증을 수행합니다.
                        val_bt = tokenize_batch(
                            val_batch,
                            input_tokenizer,
                            output_tokenizer,
                            max_input_length=tokenizer_config.max_input_length,
                            max_output_length=tokenizer_config.max_output_length,
                        )
                        val_src = val_bt.src.to(device)
                        val_mask = val_src.eq(input_tokenizer.pad_id)
                        logits = model(val_src, val_mask)
                        pred_digits = torch.argmax(logits, dim=-1).cpu().tolist()
                        for seq in pred_digits:
                            preds_all.append(digits_to_string(seq))
                        # 검증 데이터셋의 정답, 입력을 리스트에 추가합니다.
                        targets_all.extend(val_batch["target_text"])
                        inputs_all.extend(val_batch["input_text"])

                    # 검증 데이터셋의 예측, 정답을 사용하여 성능 지표를 계산합니다.
                    em_batch = compute_metrics(preds_all, targets_all)

                    # 진행바에도 성능을 표시합니다.
                    pbar.write(f"[valid {step}] EM={em_batch['EM']:.3f} TES={em_batch['TES']:.3f}")
                    pbar.set_postfix(
                        EM=f"{em_batch['EM']:.3f}",
                        TES=f"{em_batch['TES']:.3f}",
                    )
                    
                    pbar.refresh()

                    # 최고 성능 갱신 시 전체 체크포인트 저장
                    if train_config.save_best_path is not None:
                        current_em = float(em_batch.get("EM", -1.0))
                        if current_em > best_em:
                            best_em = current_em
                            # 세 config를 dict로 변환하여 저장
                            ckpt = {
                                "model_state": model.state_dict(),
                                "optim_state": optim.state_dict(),
                                "step": step,
                                "train_config": train_config.__dict__,  # 학습 설정 저장
                                "model_config": model_config.__dict__,  # 모델 설정 저장
                                "tokenizer_config": tokenizer_config.__dict__,  # 토크나이저 설정 저장
                            }
                            torch.save(ckpt, train_config.save_best_path)
                            pbar.write(f"New best EM={best_em:.3f} at step {step}; saved to {train_config.save_best_path}")

                    B = len(preds_all) # 검증 데이터셋의 크기
                    n_show = min(train_config.show_valid_samples, B)
                    pbar.write("Sample validation output:") # 예시로 몇 개만 보여줍니다.
                    for i in range(n_show):
                        input_str = inputs_all[i]
                        tgt = targets_all[i]
                        pred = preds_all[i]
                        ok = "OK" if pred == tgt else "ERR"
                        pbar.write(f"  [{i}] {ok} | input: {input_str} | target: {tgt} | pred: {pred}")
                model.train()  # 다시 학습 모드로

            # max_train_steps 제한이 있을 시, 제한을 다 채우면 학습을 종료합니다.
            if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
                break # 학습을 종료합니다.

            pbar.update(1) # tqdm 진행 1 step

# ======================================================================================
# 2. main 함수
# ======================================================================================
def main():
    # GPU가 있으면 GPU, 없으면 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 1) 데이터 준비
    # --------------------------------------------------------------------------
    
    # Train Dataset, 자세한 설정은 dataloader.py를 참고하세요.
    train_dataset = ArithmeticDataset(
        num_samples=500_000,
        max_depth=5,
        num_digits=(1, 5),
        seed=123,
        mode="train",
    )

    # Train DataLoader, 자세한 설정은 dataloader.py를 참고하세요.
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )

    # Validation Dataset, 자세한 설정은 dataloader.py를 참고하세요.
    val_dataset = ArithmeticDataset(
        num_samples=128,
        max_depth=5,
        num_digits=(1, 5),
        seed=999,
        mode="val",
    )
    
    # Validation DataLoader, 자세한 설정은 dataloader.py를 참고하세요.
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 2) 토크나이저 설정 준비
    # --------------------------------------------------------------------------
    # 토크나이저 설정 (별도로 관리)
    tokenizer_config = TokenizerConfig(
        input_chars=INPUT_CHARS,
        output_chars=OUTPUT_CHARS,
        add_special=True,
        max_input_length=64,
        max_output_length=128,
    )

    # --------------------------------------------------------------------------
    # 3) 토크나이저 준비
    # --------------------------------------------------------------------------
    # 입력 문자 토크나이저, 출력 문자 토크나이저를 준비합니다. 자세한 설정은 model.py를 참고하세요.
    input_tokenizer = CharTokenizer(
        tokenizer_config.input_chars if tokenizer_config.input_chars is not None else INPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )
    output_tokenizer = CharTokenizer(
        tokenizer_config.output_chars if tokenizer_config.output_chars is not None else OUTPUT_CHARS,
        add_special=tokenizer_config.add_special,
    )

    # --------------------------------------------------------------------------
    # 5) 학습 설정 준비
    # --------------------------------------------------------------------------
    # 학습 하이퍼파라미터 설정
    train_config = TrainConfig(
        max_train_steps=None,
        lr=2e-3,
        valid_every=200,
        max_gen_len=24,
        show_valid_samples=50,
        num_epochs=50,
        save_best_path="best_model.pt",
    )

    # --------------------------------------------------------------------------
    # 6) 모델 준비
    # --------------------------------------------------------------------------
    # Transformer 기반의 TinySeq2Seq 모델을 준비합니다.
    model_config = ModelConfig(
        d_model=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_rel_distance=64,
        output_length=tokenizer_config.max_output_length,
    )

    # ModelConfig의 모든 필드를 **kwargs로 전달
        # 숫자 토큰 id 리스트 생성 (0~9)
    digit_token_ids: List[int] = []
    for ch in "0123456789":
        if ch in input_tokenizer.stoi:
            digit_token_ids.append(input_tokenizer.stoi[ch])

    model = TinySeq2Seq(
        in_vocab=input_tokenizer.vocab_size,
        out_vocab=output_tokenizer.vocab_size,
        # FPE(Abacus)용 추가 인자
        max_input_length=tokenizer_config.max_input_length,  # seq_pos, max_seq_len 정렬
        max_digit_pos=tokenizer_config.max_input_length,     # 자리수 최대 길이 (여유 있게)
        digit_token_ids=digit_token_ids,                     # 어떤 토큰이 숫자인지
        **model_config.__dict__,  # 나머지는 그대로
    )


    # --------------------------------------------------------------------------
    # 6) 학습 시작
    # --------------------------------------------------------------------------

    train_loop(
        model=model,
        dataloader=train_dataloader,              # 미리 만든 DataLoader를 직접 전달합니다.
        input_tokenizer=input_tokenizer,        # (tokenize_batch 유틸리티를 위해 전달)
        output_tokenizer=output_tokenizer,
        device=device,
        val_dataloader=val_dataloader,
        train_config=train_config,  # 학습 설정 전달
        model_config=model_config,  # 모델 설정 전달
        tokenizer_config=tokenizer_config,  # 토크나이저 설정 전달
    )

    # --------------------------------------------------------------------------
    # 6) 학습 끝나면 모델 저장
    # --------------------------------------------------------------------------
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")


# python train.py로 실행했을 때만 main()을 돌게 합니다.
if __name__ == "__main__":
    main()