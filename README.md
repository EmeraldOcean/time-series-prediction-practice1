🕒 Time-Series Prediction & Anomaly Detection Practice  
이 프로젝트는 UCI의 Steel Industry Energy Consumption Dataset을 활용하여 시계열 데이터 예측 및 이상 탐지(Anomaly Detection)를 실습하기 위한 공간입니다.  
LSTM과 Autoencoder 두 가지 모델 아키텍처를 구현하여 성능을 비교하고, 예측 오차를 기반으로 한 이상치 판단 로직을 포함하고 있습니다.  
<br><br>
📊 Dataset  
- Source: https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption
- Description: 대한민국 시흥시 소재 철강 공장의 에너지 소비 데이터입니다.
- Key Features: Usage_kWh, Lagging/Leading Current Reactive Power, CO2, Status 등
<br><br>
🛠 Model Architectures  
본 프로젝트에서는 시계열 데이터 처리에 특화된 두 가지 접근 방식을 사용합니다.
<br><br>
1. LSTM (Long Short-Term Memory)
- 시계열 데이터의 순차적 특성을 학습하여 다음 시점의 값을 직접 예측합니다. 과거의 트렌드를 파악하는 데 강점이 있습니다.  

2. Autoencoder
- 입력 데이터를 압축(Encoding)한 후 다시 복원(Decoding)하는 과정을 거칩니다. 정상 데이터의 패턴을 학습하여, 복원 오차가 큰 데이터를 이상치로 간주하는 방식에 사용됩니다.
<br><br>
🚀 Workflow  
프로젝트의 전체 실행 흐름은 다음과 같습니다.  
1. Data Preprocessing: 데이터 로드, 결측치 처리 및 Feature Engineering을 수행합니다.
2. Model Training: 기본적으로 **정상 데이터(Normal Data)**를 중심으로 학습을 진행하여 모델이 정상 패턴에 익숙해지도록 합니다. (필요에 따라 이상 데이터를 포함하여 학습 가능)
3. Prediction & Calculation: 학습된 모델을 통해 테스트 데이터에 대한 예측값을 도출합니다.
- 실제 값(Actual)과 예측 값(Predicted) 사이의 **MSE(Mean Squared Error)**를 계산합니다.
4. Anomaly Detection: 계산된 MSE가 사전에 설정한 **특정 임계치(Threshold)**를 초과할 경우, 해당 데이터를 '이상(Anomaly)'으로 판단합니다.
<br><br>
📂 File Structure  
- data/: 데이터셋 파일 저장 (csv)
- preprocessing.py: 데이터 로드 및 전처리 스크립트
- models/: LSTM 및 AutoEncoder 모델 정의
- train.py: 모델 학습 스크립트
- eval.py: 테스트 및 결과 시각화
<br><br>
📈 Result Visualization  
학습 결과와 임계치를 시각화하여 모델이 얼마나 정확하게 이상치를 잡아내는지 확인할 수 있습니다.
<img width="2305" height="2364" alt="confusion_matrix" src="https://github.com/user-attachments/assets/2c2d4f18-52ce-4287-8362-041fd645eb13" />
<img width="2364" height="2364" alt="roc_curve" src="https://github.com/user-attachments/assets/7644da3f-6320-484b-b353-781aa6487a33" />
<br><br>
📖 How to Use  
`config.ini` 파일에서 원하는 방식대로 설정한 후, 아래의 명령어를 입력합니다.  

```bash
# 1. 저장소 복제 및 이동
git clone https://github.com/EmeraldOcean/time-series-prediction-practice1.git
cd time-series-prediction-practice1

# 2. 코드 실행
python main.py
