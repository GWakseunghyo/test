# 내용 정리
# 각 단계별 코드의 정리
# 총 4단계로 구성
# 1~2 단계가 가장 중요할 것으로 보임

# 1단계 GRU 모델 정의 
# 이 코드에서는 class 구문을 통해 GRUNet를 정의 
# PyTorch의 nn.Module을 이용한다
# 모델은 아래 다섯 가지의 인자를 입력받는다.
#       1.입력 차원(input_dim)
#       2.은닉 상태의 차원(hidden_dim) 
#       3.출력 차원(output_dim) 
#       4.GRU 레이어의 수(n_layers)  
#       5.드롭아웃 비율(drop_prob)
# 이후 3가지 함수를 정의함으로써 GRU모델을 정의하게 된다.
# __init__ 
#       GRU 레이어와 fully connected 레이어를 정의 
#       ReLU 활성화 함수를 사용
# forward 
#       입력 데이터와 초기 은닉 상태를 받아서 GRU를 실행 
#       마지막 시간 단계의 출력을 fully connected 레이어에 통과시켜 예측값을 생성
# init_hidden
#       초기 은닉 상태를 초기화

# 2단계 trainning
# 데이터를 받고 train함수를 이용하여 모델의 트레이닝을 진행
# 이때 모델의 구조와 하이퍼파라미터를 정의 
# MSE(손실 함수)와 Adam(옵티마이저)를 정의 =>최대한 오차(손실)이 적어지는 방향으로 교육한다는 의미
# 이후 구간을 정해서 데이터를 나누고 모델을 훈련 및 손실 계산

# 3단계 evaluate
# evaluate 함수를 이용하여 훈련된 모델을 테스트 데이터셋에 적용하고 예측값을 생성하여, MAE와 RMSE를 계산
# 예측값과 실제값의 비교 => 아마 비교하는 함수를 따로 적용해야할 듯

# 4단계 plot
# 테스트 데이터와 GRU모델 간의 데이터를 plot하여 비교

#----------------------------------------------------------------------------------------------



#현재 폴더는 이미 가상환경을 activate한 상태 => 명령 프롬프트를 이용하여 구동, 관련 함수는 메모장 참고
#참고내용
#----------------------------------------------------------------------------------------------
#내가 이해할 수 있는 전체적인 가상환경 구축 과정
#1. 우선 가상 환경을 구축할 수 있도록 초기 셋팅을 해준다.
#  해당 코드는 다음과 같다. 단 인때 경로는 그냥 내가 마음대로 해놓음
#   C:\project>python -m venv example
#   C:\project>cd example

#2. 이제는 그 경로에서 파일을 만들고 내가 이 파일에서 가상환경을 실행하겠다고 선언을 해야한다. 
#PS C:\Users\곽승효\example\Scripts> activate

#-----------------------------------------------------------------------------------------------


import scipy.io #사이파이에서 기본적인 통계함수 모듈을 가져온다

import time # 시간에 관련한 모듈을 불러온다
import math # 수학 공식에 관련한 모듈을 불러온다

import torch # PyTorch 라이브러리를 불러온다.
import torch.nn as nn # pytorch에서 신경망 함수를 불러온다 
from torch.utils.data import TensorDataset, DataLoader # PyTorch에서 데이터셋과 데이터 로더를 가져온다.

import pandas as pd # pandas라이브러리(데이터를 조작하고 분석용)를 가져온다
import numpy as np #numpy(다차원 배열 핸들링)를 가져온다
import matplotlib.pyplot as plt #데이터 시각화를 위한 라이브러리인 matplotlib의 pyplot을 가져온다.
from sklearn.preprocessing import MinMaxScaler # 데이터 전처리를 위한 MinMaxScaler를 가져온다.

mat = scipy.io.loadmat('./datasets/Oxford_Battery_Degradation_Dataset_1.mat') 
#데이터 셋을 불러오는 함수

###################
# 내가 수정할 부분의 중심이 된다.
# 방향은 .mat 형식의 파일을 엑셀에서도 열 수 있도록 => 관련 코드를 따로 불러오는게 메인이 될 듯 하다.
# 아마 단순하게 data_set = read.data()이런식으로 데이터를 읽어와서 저장하는 방식으로 진행할 듯.
# ex
# excel_file = './datasets/Oxford_Battery_Degradation_Dataset_1.mat'식으로 경로를 지정하고
# df = pd.read_excel(excel_file) 데이터 프레임으로 읽기

###################

mat.keys()
# => 이 함수는 엑셀로 치면 필드명만을 가져와서 사용한다는 의미로 보인다.
# => 결국에는 아마 이 뒤의 함수에서 단순하게 필드 명을 []를 이용해 가져오든
# 아니면 필드명을 따로 변수로 지정해서 엑셀 파일의 그 행 데이터만 저장을 하는 방식을 이용하는 것이 좋을 듯 하다.  
# R studio 에서 사용하는 방식을 이용할 듯
# ex) current  <= excel 시트명[1,,]식으로 따로 빼오는 방법을 찾아봐야 할 듯

dict_keys(['__header__', '__version__', '__globals__', 'Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6', 'Cell7', 'Cell8'])
# => 아마 매트랩 파일 정보를 뽑아 내는 듯 한데, Cell1 등은 변수 명이므로 
# 나중에 내가 사용할 때는 변수명을 더 직관적으로 명칭을 정해야 할 듯 
#input으로 들어가는 데이터 이므로 아마 온도는 고정한 채, 전류 전압을 쌍으로 하여금 넣으면 될 듯 하다.

input_data = [
    [[], []],  # cell 1, [0, 1, 2, ... 82], [730, 729, ... value for q, ... , 530]
    [[], []],
    [[], []],
    [[], []],
    [[], []],
    [[], []],
    [[], []],
    [[], []]
]

# 데이터의 입력을 2개의 쌍으로 만들기 위해 넣은 구조 



CELL_SIZE = [83, 78, 82, 52, 49, 51, 82, 82] # number of the cycles for each cell
for i in range(0, 8): #breaks down dataset to return the values that we need
    cell_num = "Cell{}".format(i + 1)
    for j in range(0, CELL_SIZE[i]):
        cyc_num = "cyc{:04d}".format(j * 100) # if j=47, cyc_num = "cyc4700"; if j=1, cyc_num = "cyc0100"
                                              #cyc{:04d}는 0000의 형태를 의미 
        try:
            curr = mat[cell_num][0][cyc_num][0][0]["C1ch"][0][0]['q'][0][-1][0]
        except ValueError:
            curr = float("NaN")
        input_data[i][0].append(j)
        input_data[i][1].append(curr)

#위 내용은 데이터를 넣는 과정에서 인덱스 처리를 해서 범위별로 데이터를 넣는 것으로 보인다       

train_x = []
train_y = []

#학습할 데이터를 저장하고자 제일 처음에 미리 구조를 만들고 초기와 해놓은 것


#test_x is a 2d dataset


#아래 두 개의 변수 x,y는 test데이터를 저장해놓기 위한 구조 
test_x = [ 
    [],
    [],
    [],
    []
]
#test_y is a 2d dataset
test_y = [
    [],
    [],
    [],
    []
]

#아래 과정은 데이터 전처리 과정을 수행

for i in range(0, 8):

    df_input = pd.DataFrame(input_data[i]).transpose()   #데이터를 데이터 프레임 형태로 재구성
    df_input = df_input.rename(columns={0: "cycle", 1: "q_value"})  #데이터 프레임에서 명칭 바꾸기
    df_input['interpolate_time'] = df_input['q_value'].interpolate(option='time') #time interpolate preprocessing of cells 1 to 8
    df_input['soh'] = df_input['interpolate_time'] / 740

    #interpolate_time(q_value를 시간에 따라 보관),soh에 대한 항목을 더 추가

    df_input = df_input.drop("cycle", axis=1)
    df_input = df_input.drop("q_value", axis=1)
    df_input = df_input.drop("interpolate_time", axis=1)
    #위 과정은 데이터의 labeling에서 사용된 필드명이 데이터 셋에 안들어가게 만들기 위해 필드명을 제거하는 과정


    # normalisation
    #데이터의 정규화 과정 => 0과 1 사이의 값을 데이터의 범위를 지정한다
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    #위와 같이 따로 정의하는 것은 객체 정의를 위한 과정
    data = sc.fit_transform(df_input.values)
    label_sc.fit(df_input.iloc[:, 0].values.reshape(-1, 1)) #MinMaxScaler preprocessing of cells 1 to 8

    # windowing of time series
    lookback = 10 #data windowing value
    inputs = np.zeros((len(data) - lookback, lookback, df_input.shape[1]))
    labels = np.zeros(len(data) - lookback)

    for j in range(lookback, len(data)):
        inputs[j - lookback] = data[j - lookback:j]
        labels[j - lookback] = data[j, 0]
    inputs = inputs.reshape(-1, lookback, df_input.shape[1])
    labels = labels.reshape(-1, 1)

    if (i < 4):
        if len(train_x) == 0:
            train_x = inputs[:]
            train_y = labels[:]
        else:
            train_x = np.concatenate((train_x, inputs[:]))
            train_y = np.concatenate((train_y, labels[:]))
    else:
        test_x[i - 4] = inputs
        test_y[i - 4] = labels

np.array(train_x).shape
(255, 10, 1)
#첫 번째 차원은 데이터 샘플의 수(255)
#두 번째 차원은 시퀀스 길이(10) 
#세 번째 차원은 입력 특성의 수(1)

np.array(train_y).shape
(255, 1)
np.array(test_x[0]).shape # test_x = [[Cell5], [Cell6], [Cell7], [Cell8]]
(39, 10, 1)
np.array(test_y[0]).shape
(39, 1)


#아래 코드는 쿠다를 이용하여 gpu를 따로 더 사용할 수 있는가를 판단하고 
#Gpu를 더 가속시키기 위해 사용하는 코드
batch_size = 32
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.


#신경망 모델을 정의하고 학습하는 함수를 정의  
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__() 
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

##############################

#아래 항목은 lstm에 대한 module을 형성 
#GRU조사 입장에서는 크게 필요가 없으나 
#코드 상에서 문제가 생길 수도 있으니, 나중에 제거하는 걸로 

##############################

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

################################



def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=100, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2

    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    #우리는 GRU만 다루므로 그냥 if문 쓰지 말고 model = GRUnet이렇게 쓰면 될 듯
    
    model.to(device)

    print("----------------------------------------")
    print("The", model_type, "has the layers as: ")
    print(model)
    print("----------------------------------------")

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.process_time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            # 이 부분도 아마 lstm을 고려해서 넣은 항목이므로 빼면 된다.

            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                       len(train_loader),
                                                                                       avg_loss / counter))

        current_time = time.process_time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model
def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.process_time()

    inp = torch.from_numpy(np.array(test_x))
    labs = torch.from_numpy(np.array(test_y))
    h = model.init_hidden(inp.shape[0])
    out, h = model(inp.to(device).float(), h)
    outputs.append(label_sc.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
    targets.append(label_sc.inverse_transform(labs.numpy()).reshape(-1))

    print("Evaluation Time: {}".format(str(time.process_time() - start_time)))
    MAE = 0
    RMSE = 0
    for i in range(len(outputs)):
        MAE += np.mean(abs(outputs[i] - targets[i])) / len(outputs)
        RMSE += np.mean((outputs[i] - targets[i]) ** 2) / len(outputs)
    RMSE = math.sqrt(RMSE)
    print("MAE: {}".format(MAE))
    print("RMSE: {}".format(RMSE))
    return outputs, targets, MAE, RMSE
lr = 0.001
gru_model = train(train_loader, lr, model_type="GRU")


#~곳에다가 저장을 하겠다.
torch.save(gru_model.state_dict(), "./saved_models/gru_model.pt")

#plot 하는 항목들을 정리
for i in range(0, 4): #plot both graphs
    print("GRU Evaluation for Cell {}".format(i + 5))
    gru_outputs, targets, gru_MAE, gru_RMSE = evaluate(gru_model, test_x[i], test_y[i], label_sc)
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.plot(gru_outputs[0][-100:], "-o", color="g", label="Predicted")
    plt.plot(targets[0][-100:], color="b", label="Actual")
    plt.xlabel('Cycle')
    plt.ylabel('SOH')
    plt.title("Cell {}".format(i + 5))
    plt.legend()
    plt.show()

