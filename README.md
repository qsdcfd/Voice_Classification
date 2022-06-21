
# Voice_Classification



## 주관: Dacon

<br>

![image](https://user-images.githubusercontent.com/86671456/173496078-c005565d-3399-4f67-8c3a-bad2badd8658.png)


## Abstract

| 분석명 |  
|:-----:|
| 소리 분류 |

|  소스 데이터 |     데이터 입수 난이도    |      분석방법     |
|:------------------:| -----|:---------------:|
|train| 하| mfcc, mel, CNN  |

|  분석 적용 난이도  |     분석적용 난이도 사유    |      분석주기     | 분석결과 검증 Owner|
|:-----:| --------------------------------------- |:---------------:|----------------|
|상 |음성데이터를 처음 접하였다   |Daily  | Dacon |



<br>

### Machine Learning Project 

---

|  프로젝트 순서 |     Point    | 세부 내용 |  
|:------------------:| -----|------|
|문제 정의|해결할 점, 찾아내야할 점 |9가지 종류의 소리를 분류하기|
|데이터 수집|공개 데이터, 자체 수집, 제공된 데이터 |train, test, submission|   
|데이터 전처리|문제에 따라서 처리해야할 방향 설정 |주파수와 라벨링된 정답과의 관계파악하기
|Feature Engineering|모델 선정 혹은 평가 지표에 큰 영향|주파수를 이미지로 만들기|
|연관 데이터 추가|추가 수집 | 훈련, test  |추가 데이터를 수집하진 않았다.|
|알고리즘 선택| 기본적, 현대적|RF, LDA, CNN|   
|모델 학습|하이퍼파라미터,데이터 나누기 |Grident Search CV, 7:3, 8:2, 9:1 방식으로 나눠보는 실험 |
|모델 평가|확률  |Accuracy:0.6(70위) |
|모델 성능 향상|성능 지표, 하이퍼파라미터, 데이터 리터러시 재수정 |Accuracy:1.0 (ToP6)  |

<br>

### Basic information

**공식기간: 2022.06.13 ~ 2022.06.24**


- 인원:이세현
- 직책: 대회 참가자
- 데이터: 음성데이터(wav)
- 주 역할: Data literacy, Data Preprocessing, Model Selection, Hyperparameter-tuning
- 협업장소: Github, GoogleMeet
- 소통: Slack, Notion,Git project, Google OS
- 저장소: Github, Google Drive
- 개발환경: Visual studio code, Juypter Notebook, colab
- 언어 :python 3.8.x
- 라이브러리:Numpy,Pandas, Scikit-learn 1.1.x, Pytorch, Tensorflow, Keras
- 시각화 라이브러리: Seaborn, Matplotlib, Plot,Plotly  
- 시각화 도구: Tableau, GA

<br>

#### 파일 설명

- feat: 기능 개발 관련
- fix: 오류 개선 혹은 버그 패치
- docs: 문서화 작업
- test: test 관련
- conf: 환경설정 관련
- build: 데이터 집산
- Definition: 프로젝트의 전반적인 문제 정의 및 내용 설정
- Data: 전처리 파일 및 모델링을 위한 파일
- models: 여러 모델들의 집합
- src :scripts
