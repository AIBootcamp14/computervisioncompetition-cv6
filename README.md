# Title (Please modify the title)
## Team

| ![김시진](https://avatars.githubusercontent.com/u/46598332?v=4) | ![신준엽](https://avatars.githubusercontent.com/u/180160571?v=4) | ![이가은](https://avatars.githubusercontent.com/u/217889143?v=4) | ![이건희](https://avatars.githubusercontent.com/u/213379929?v=4) | ![이  찬](https://avatars.githubusercontent.com/u/100181857?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김시진](https://github.com/kimsijin33)             |            [신준엽](https://github.com/Shin-junyeob)             |            [이가은](https://github.com/kkaeunii)             |            [이건희](https://github.com/GH-Lee33)             |            [이  찬](https://github.com/SKKULEE)             |
|                             팀장, EDA / model 실험                             |                             EDA / model 실험                             |                            model 실험                             |                            model 실험                             |                            EDA, Augmentation                             |

# 🖼️ Document Image Classification

이 프로젝트는 문서 이미지 데이터를 분류하는 Computer Vision 모델을 학습하여, 예측 결과를 `.csv` 파일로 제출하는 것을 목표로 합니다.

## 📂 Repository Structure
- `main.ipynb` : 데이터 전처리, 모델 학습, 추론 파이프라인 코드
- `submission.csv` : 최종 제출용 예측 결과 파일
- `requirements.txt` : 실행 환경 설정을 위한 패키지 목록

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

## 🚀 How to Run
1. VScode 상에서 진행:
- main.ipynb 파일을 셀단위로 실행

2. 학습 및 추론을 통해 예측 결과(`.csv`) 생성

## 📊 Example Output
제출 파일(`submission.csv`) 예시:
```csv
ID,Target
0001,3
0002,7
0003,12
...
```

## 📌 Result
- Public Leaderboard Score: **0.9579**
- Private Leaderboard Score: **0.9518**

## 🙌 Acknowledgement
- Dataset: 제공된 대회 데이터셋
- Frameworks: PyTorch, timm, Albumentations
