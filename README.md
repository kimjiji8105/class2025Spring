**2025511030**
---

# 악성 댓글 탐지 모델 비교 실험

## 과제 개요

본 프로젝트는 자연어처리(Natural Language Processing) 과목의 과제로 수행되었습니다.  
온라인 서비스 상에서 반복적으로 발생하는 악성 댓글을 효과적으로 탐지하기 위한 모델을 비교하고,  
문맥 이해 능력이 성능에 어떤 영향을 미치는지 분석하는 것을 목표로 합니다.

## 실험 목적

기존 TF-IDF 기반 분류 모델은 단어 빈도 중심으로 학습되기 때문에 문맥 기반 의미를 반영하기 어렵습니다.  
이를 개선하고자 사전학습 언어 모델인 KoBERT를 파인튜닝하여 적용하고, 기존 모델과 성능을 비교하였습니다.  
모델 성능 비교는 문제성 문장(혐오 표현 또는 악플)에 대한 F1-score를 중심으로 평가하였습니다.

## 데이터셋

- `korean_unsmile_dataset` (총 18,742건)  
- `hatescore-korean-hate-speech` (총 11,108건)  

총 29,850건의 문장을 문제성(1)과 정상(0)의 이진 라벨로 통합하였으며,  
Stratified 방식으로 학습, 검증, 테스트 세트를 8:1:1 비율로 분할하였습니다.

## 모델 구성

- **Baseline**: TF-IDF (1~2gram) + Logistic RegressionCV  
- **Fine-tuned KoBERT**: `monologg/kobert` 기반, LoRA(r=8, α=16) 적용한 부분 파인튜닝

## 실험 환경 및 실행

본 프로젝트는 Google Colab 환경에서 수행되었으며, 다음과 같은 패키지 설치 명령어를 사용하였습니다:

```bash
!pip install -q transformers evaluate accelerate datasets koco peft==0.13.2
```

단일 실행 파일로 학습부터 평가까지 수행되며, 별도 프로젝트 구조 없이 실행 가능합니다.

## 성능 비교 요약

| 모델     | F1-score(문제성) | Accuracy |
|----------|------------------|----------|
| Baseline | 0.8395           | 0.8151   |
| KoBERT   | 0.8634           | 0.8630   |

KoBERT 모델은 Precision과 Recall의 균형이 우수하며,  
정상 문장에 대한 오탐률을 크게 줄여 전반적으로 안정적인 성능을 보였습니다.  
문맥 이해가 필요한 문장에서는 기존 모델보다 효과적인 탐지가 가능했습니다.

## 주요 인사이트

- TF-IDF 모델은 명시적인 욕설이나 혐오 표현 탐지에는 유리하나,  
  문맥이 복합적인 문장에서는 정상 문장을 문제성으로 오분류하는 경향이 있었습니다.  
- KoBERT는 다양한 표현 유형에 대해 균형 잡힌 탐지 성능을 보였으며,  
  반어적 표현이나 간접적인 문장에 대해서도 상대적으로 우수한 판단을 수행하였습니다.  
- 데이터 구성 특성상 TF-IDF 모델도 일정 수준 이상의 성능을 보였으나,  
  실제 서비스 환경에서는 KoBERT의 문맥 이해 능력이 더욱 효과적일 수 있습니다.

## 참고 자료

- KoBERT: https://github.com/SKTBrain/KoBERT  
- korean_unsmile_dataset: https://github.com/naver/nlp-data  
- hatescore-korean-hate-speech: https://huggingface.co/datasets/kocohub/hatescore-korean-hate-speech
