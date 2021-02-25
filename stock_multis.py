#타겟 종목별로 학습 버전(n : 1), stcok_mcroll.py 혹은 stcok_mcroll2.py로 수집해야 함

import mempute as mp
from netshell import NetShell
import numpy as np
import pandas as pd
import  sys
import datetime
import matplotlib.pyplot as plt

# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원

# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def rev_minmax(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

#python stock_multis.py exec_case stride gap out_length ycol i_stock train_rate epoch
exec_case = sys.argv[1]
stride = int(sys.argv[2])
gap = int(sys.argv[3])
out_length = int(sys.argv[4])
ycol = int(sys.argv[5])
i_stock = sys.argv[6]

prev_length = 16          #32
seq_length = 0  # by gate에 적재할 입력 시퀀스의 길이(시계열데이터 입력 개수)
by_length = seq_length + out_length + 1

if int(exec_case) < 6:
    train_rate = float(sys.argv[7])
if stride == 0: stride = out_length #out_length는 스트라이드 역활
# 데이터를 로딩한다.
STOCK_STORE = './stock_data3'
stock_path = STOCK_STORE + '/' + i_stock
in_df = pd.read_csv(stock_path + '/stock_dsor.csv', header=None) #판다스이용 csv파일 로딩
tar_df = pd.read_csv(stock_path + '/stock_dtar.csv', header=None) #판다스이용 csv파일 로딩

stock_din = in_df.values # 금액&거래량 문자열을 부동소수점형으로 변환한다
stock_dtar = tar_df.values # 금액&거래량 문자열을 부동소수점형으로 변환한다

data_len = len(stock_dtar) #one stock data size
n_in_stock = int(len(stock_din) / data_len) #stock_din - [n_in_stock * data_len, feat_sz]
feat_sz = stock_din.shape[1]
out_feat = 1
close_icol = 4
stock_data = np.concatenate((stock_din, stock_dtar), axis=0) 

tcr = mp.tracer(0, "msstock")
mp.reposet(tcr, stock_path)
#mp.npset(tcr, 10000)

# 데이터 전처리
# 가격과 거래량 수치의 차이가 많아나서 각각 별도로 정규화한다

# 가격형태 데이터들을 정규화한다
# ['Open','High','Low','Close','Adj Close','Volume']에서 'Adj Close'까지 취함
# 마지막 열 Volume를 제외한 모든 열
price = stock_data[:,:-2]
p_close = stock_data[:,-1:]
price = np.concatenate((price, p_close), axis=1) # axis=1, 세로로 합친다
print(price)
norm_price = min_max_scaling(price) # 가격형태 데이터 정규화 처리
print("price.shape: ", price.shape)
print("price[0]: ", price[0])
print("norm_price[0]: ", norm_price[0])
print("="*100) # 화면상 구분용
# 거래량형태 데이터를 정규화한다
# ['Open','High','Low','Close','Adj Close','Volume']에서 마지막 'Volume'만 취함
# [:,-1]이 아닌 [:,-1:]이므로 주의하자! 스칼라가아닌 벡터값 산출해야만 쉽게 병합 가능
volume = stock_data[:,-2:-1]
print(volume)
norm_volume = min_max_scaling(volume) # 거래량형태 데이터 정규화 처리
print("volume.shape: ", volume.shape)
print("volume[0]: ", volume[0])
print("norm_volume[0]: ", norm_volume[0])
print("="*100) # 화면상 구분용
# 행은 그대로 두고 열을 우측에 붙여 합친다
norm_stock = np.concatenate((norm_price, norm_volume), axis=1) # axis=1, 세로로 합친다
print(norm_stock.shape)
norm_stock = norm_stock.reshape((n_in_stock + 1, data_len, feat_sz))
print(norm_stock.shape)
x_norm = norm_stock[:-1] #[n_in_stock, data_len, feat_sz]
y_norm = norm_stock[-1] #[data_len, feat_sz]
datax = [] # 입력으로 사용될 Sequence Data
datay = [] # 출력(타켓)으로 사용
if data_len != len(y_norm): exit()
n = data_len - (prev_length + gap + out_length)
i = n % stride #나머지를 시작쪽에서 제외
while i <= n:
    xl = []
    for j in range(n_in_stock):
        if feat_sz == 1: x = x_norm[j, i : i+prev_length, [close_icol]] #[prev_length, 1]
        else: x = x_norm[j, i : i+prev_length] #[prev_length, feat_sz]
        xl.append(x)
    xl = np.array(xl) #[n_in_stock, prev_length, featsz|1]
    if feat_sz == 1: xl = xl.reshape((n_in_stock*prev_length, 1)) #[n_in_stock*prev_length, 1]
    else: xl = xl.reshape((n_in_stock*prev_length, feat_sz)) #[n_in_stock*prev_length, featsz]
    if out_feat == 1: 
        y = y_norm[i + prev_length + gap: i + prev_length + gap + out_length, [close_icol]] #[out_length, 1]다음 나타날 주가(정답)
    else: 
        y = y_norm[i + prev_length + gap: i + prev_length + gap + out_length] #[out_length, out_feat] 다음 나타날 주가(정답)
    datax.append(xl) # dataX 리스트에 추가
    datay.append(y) # dataY 리스트에 추가
    i = i + stride

length = len(datay)
datax = np.array(datax) #[length, n_in_stock*prev_length, featsz|1]
datay = np.array(datay) #[length, out_length, out_feat|1]
"""
for i, v in enumerate(rev_minmax(price, datax)):#volume은 오리진이 다르디만 무시한다.
    if i < 10 or i > length - 10: print(i*stride+start, ": ", v)
print("="*100) # 화면상 구분용

for i, v in enumerate(rev_minmax(price, datay)): 
    if i < 10 or i > length - 10: print(i*stride+start+prev_length, ": ", v)
print("="*100) # 화면상 구분용
"""
"""
if data_len % out_length:
    data_len = (data_len % out_length) #margin을 앞쪽에서 제외한다.
    norm_stock = norm_stock[:, data_len:]
print('data length: ', data_len, 'out len: ', out_length, 'cut len: ', norm_stock.shape[1])
norm_stock = norm_stock.reshape((n_in_stock + 1, int(data_len / out_length), out_length, feat_sz))
print("norm_stock.shape: ", norm_stock.shape)
x_norm = norm_stock[:-1, :-1] #입력 주식은 끝을 버린다.
print("x_norm.shape: ", x_norm.shape)
y_norm = norm_stock[-1:, -1:] #타겟 주식은 시작을 버린다.
print("y_norm.shape: ", y_norm.shape)
norm_stock = np.concatenate((x_norm, y_norm), axis=0)
print("norm_stock.shape: ", norm_stock.shape)

n_batch = int(data_len / out_length) - 1
if n_batch != norm_stock.shape[1]: 
    print('batch error')
    exit()
stock_fx = mp.flux(tcr, norm_stock, mp.constant) #[n_stock, n_batch, len_seq, feat], n_stock - 타겟주식 포함
uns_stock = mp.unstack(stock_fx, 1)
data_fair = []
for s in uns_stock:
    data_fair.append(mp.eval(s))
data_fair = np.array(data_fair) #[n_batch, n_stock, len_seq, feat]

for i, v in enumerate(data_fair):
    if i < 60 or i > len(data_fair) - 60: print(i, ": ", v)
print("="*100) # 화면상 구분용

data_x = data_fair[:, :-1]
data_x = data_fair.reshape((n_batch, n_in_stock * out_length, feat_sz)) #[n_batch, n_in_stock*len_seq, feat]
data_y = data_fair[:, -1] #[n_batch, len_seq, feat]
data_y = data_y[:, [-2]] # 타켓은 주식 종가이다

for i, v in enumerate(data_y): 
    if i < 60 or i > len(data_y) - 60: print(i, ": ", v)
print("="*100) # 화면상 구분용
length = len(data_y)
"""
def shuffle(x: np.array, y: np.array):
    N = x.shape[0]
    indices = np.random.permutation(np.arange(N))
    x = x[indices]
    y = y[indices]
    return x, y
print('stock data size: ', data_len)
print('num input stock: ', n_in_stock)

unit_sz = 32
mp.traceopt(tcr, 1, 1)
mp.traceopt(tcr, 0, 4)
mp.traceopt(tcr, 8, 1)
mp.traceopt(tcr, 60, -1)
#mp.traceopt(tcr, 62, 0)
#mp.traceopt(tcr, 4, 16)
#mp.traceopt(tcr, 63, 0)
#mp.traceopt(tcr, 65, 0)
mp.traceopt(tcr, 67, 20)
mp.traceopt(tcr, 53, 1) #1d
mp.traceopt(tcr, 69, 777)
mp.traceopt(tcr, 70, 1)
mp.traceopt(tcr, 43, 32)
mp.traceopt(tcr, 54, 0.001)
mp.traceopt(tcr, 75, 2)
mp.traceopt(tcr, 76, 1)
cell = NetShell(tcr, psz = datax.shape[1], bsz=by_length, ysz=out_length + 1, latent_sz=32, indisc = 0, ydisc = 0, embedim = 0, feat = out_feat, prevfeat = feat_sz)
lv = 1
vloss_check = 0.000001
vloss_check2 = 0.0003
if exec_case is '1': #python stock_multis.py exec_case stride gap out_length ycol i_stock train_rate epoch
    epoch = int(sys.argv[8]) #(1 1 0 7 6 45 0.7 1000)
    start = 0
    # 학습용/테스트용 데이터 생성
    # 전체 70%를 학습용 데이터로 사용
    train_length = int(length * train_rate)
    #test_length = int(length * train_rate)
    # 나머지(30%)를 테스트용 데이터로 사용
    test_length = length - train_length
    # 데이터를 잘라 학습용 데이터 생성
    #trainX = np.array(datax[0:train_length])
    #trainY = np.array(datay[0:train_length])
    if train_length < unit_sz: batch_size = train_length
    else: batch_size = unit_sz
    datax2 = datax[0:train_length]
    datay2 = datay[0:train_length]
    print('batch size: ', batch_size, 'train size: ', train_length, 'test size: ', test_length)
    print('press any key train start')
    input()
    start = train_length % batch_size #나머지를 시작쪽에서 제외
    rv = 1
    ep = 0
    max_v = 100000
    while ep < epoch:
        i = start
        datax2, datay2 = shuffle(datax2, datay2)
        while i < train_length:
            x = datax2[i : i+batch_size] #[batch_size, n_in_stock*prev_length, featsz|1]
            y = datay2[i : i+batch_size] #[batch_size, out_length, out_feat|1]
            #print(i, ' : ', rev_minmax(price, x[0]), " -> ", rev_minmax(price, y[0]))
            total_loss, rv = cell.train(x, None, y) #rv는 dyanagen을 실행될때 종료 여부 리턴
            #mp.printo(total_loss)
            if ep == 0: mp.printo(total_loss)
            if rv == 0:#정확도에 의한 종료, 옵션으로 ignore, 외부에서 종료(에포크)
                break
            i += batch_size
        if ep > 0: mp.printo(total_loss)
        if mp.eval(total_loss) > 0: ep += 1 #최종 커플isle 학습이 시작되야 에포크 증가
        if ((ep+1) % 100 == 0) or (ep == epoch-1): 
            accu, _, train_predict, train_y = cell.accuracy(datax[start:train_length], None, datay[start:train_length], batch_size)
            print('epoch:', ep, 'train accuracy:', accu) #평균제곱오차는 낮을수록 정확도 높음
            accu2, _, test_predict, test_y = cell.accuracy(datax[train_length:length], None, datay[train_length:length], batch_size)
            print('test accuracy:', accu2) #평균제곱오차는 낮을수록 정확도 높음
            if accu2 <= max_v and ep > 700:
                print('save accuracy', accu2)
                max_v = accu2
                cell.recording()
            if accu2 < vloss_check2 and ep > 700:
                break
        #if lv < vloss_check:
        #        break
        if rv == 0:
            break
    #cell.recording()
    if out_feat != 1:
        train_predict = train_predict[:,:,[close_icol]]
        train_y = train_y[:,:,[close_icol]]
        test_predict = test_predict[:,:,[close_icol]]
        test_y = test_y[:,:,[close_icol]]
    print(train_predict.shape)
    print(train_y.shape)
    print('train complete & press any key prediction')
    input()
    """
    _train_predict = train_predict.reshape((train_predict.shape[0], train_predict.shape[1]))
    _train_y = train_y.reshape((train_y.shape[0], train_y.shape[1]))
    for (r, p) in zip(_train_y, _train_predict):
        print("="*100)
        print(r)
        print("-"*100)
        print(p)
        print("="*100)
    """
    train_predict = train_predict[:,ycol:ycol+stride] #out_length중 ycol ~ ycol+stride 것으로 출력
    train_y = train_y[:,ycol:ycol+stride]
    _train_predict = train_predict.reshape(-1, 1)
    _train_y = train_y.reshape(-1, 1)
    plt.figure(1)
    plt.plot(_train_y, 'r')
    plt.plot(_train_predict, 'b')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()
    """
    _test_predict = test_predict.reshape((test_predict.shape[0], test_predict.shape[1]))
    _test_y = test_y.reshape((test_y.shape[0], test_y.shape[1]))
    for (r, p) in zip(_test_y, _test_predict):
        print("="*100)
        print(r)
        print("-"*100)
        print(p)
        print("="*100)
    """
    test_predict = test_predict[:,ycol:ycol+stride] #out_length중 ycol ~ ycol+stride 것으로 출력
    test_y = test_y[:,ycol:ycol+stride]
    _test_predict = test_predict.reshape(-1, 1)
    _test_y = test_y.reshape(-1, 1)
    plt.figure(2)
    plt.plot(_test_y, color='r', marker='o', label='right', linestyle='--')
    plt.plot(_test_predict, color='b', marker='v', label='pred', linestyle='-.')
    #plt.plot(_test_y, 'r')
    #plt.plot(_test_predict, 'b')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

    accu, _, last_predict, last_y = cell.accuracy(datax[-1:], None, datay[-1:], 0)
    print('epoch', ep, 'train accuracy: ', accu)
    if out_feat != 1:
        last_predict = last_predict[:,:,[close_icol]]
        last_y = last_y[:,:,[close_icol]]
    _last_predict = last_predict.reshape(-1, 1)
    _last_y = last_y.reshape(-1, 1)
    plt.figure(3)
    plt.plot(_last_y, color='r', marker='o', label='right', linestyle='--')
    plt.plot(_last_predict, color='b', marker='v', label='pred', linestyle='-.')
    #plt.plot(_last_y, 'r')
    #plt.plot(_last_predict, 'b')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

elif exec_case is '2': #python stock_multis.py exec_case stride gap out_length ycol i_stock train_rate
    train_length = int(length * train_rate) #(2 1 0 7 6 3 0.7)
    if train_length < unit_sz: batch_size = train_length
    else: batch_size = unit_sz
    accu, _, train_predict, train_y = cell.accuracy(datax[0:train_length], None, datay[0:train_length], batch_size)
    print('test accuracy: ', accu)
    train_predict = train_predict[:,ycol:ycol+stride] #out_length중 ycol ~ ycol+stride 것으로 출력
    train_y = train_y[:,ycol:ycol+stride]
    _train_predict = train_predict.reshape(-1, 1)
    _train_y = train_y.reshape(-1, 1)
    plt.figure(2)
    plt.plot(_train_y, 'r')
    plt.plot(_train_predict, 'b')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

elif exec_case is '3': #python stock_multis.py exec_case stride gap out_length ycol i_stock train_rate 
    train_length = int(length * train_rate) #( 3 1 0 7 6 3 0.7)
    if train_length < unit_sz: batch_size = train_length
    else: batch_size = unit_sz
    accu, _, test_predict, test_y = cell.accuracy(datax[train_length:length], None, datay[train_length:length], batch_size)
    print('test accuracy: ', accu)
    test_predict = test_predict[:,ycol:ycol+stride] #out_length중 ycol ~ ycol+stride 것으로 출력
    test_y = test_y[:,ycol:ycol+stride]
    _test_predict = test_predict.reshape(-1, 1)
    _test_y = test_y.reshape(-1, 1)
    plt.figure(2)
    plt.plot(_test_y, color='r', marker='o', label='right', linestyle='--')
    plt.plot(_test_predict, color='b', marker='v', label='pred', linestyle='-.')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

elif exec_case is '4': #python stock_multis.py exec_case stride gap out_length ycol i_stock train_rate beg end
    beg = int(sys.argv[8])
    end = int(sys.argv[9]) 
    if end - beg < unit_sz: batch_size = end - beg
    else: batch_size = unit_sz
    train_length = int(length * train_rate)
    accu, _, last_predict, last_y = cell.accuracy(datax[train_length+beg:train_length+end], None, datay[train_length+beg:train_length+end], batch_size)
    print('train accuracy: ', accu)
    if out_feat != 1:
        last_predict = last_predict[:,:,[close_icol]]
        last_y = last_y[:,:,[close_icol]]
    last_predict = last_predict[:,ycol:ycol+stride] #out_length중 ycol ~ ycol+stride 것으로 출력
    last_y = last_y[:,ycol:ycol+stride]
    _last_predict = last_predict.reshape(-1, 1)
    _last_y = last_y.reshape(-1, 1)
    plt.figure(3)
    plt.plot(_last_y, color='r', marker='o', label='right', linestyle='--')
    plt.plot(_last_predict, color='b', marker='v', label='pred', linestyle='-.')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

elif exec_case is '5': #python stock_multis.py exec_case stride gap out_length ycol i_stock train_rate beg
    beg = int(sys.argv[8])
    train_length = int(length * train_rate)
    pred_res = cell.predict(datax[train_length+beg:train_length+beg + 1], None)
    if out_feat != 1:
        pred_res = pred_res[:,:,[close_icol]]
    pred_res = pred_res.reshape(-1, 1)
    pred_y = datay[train_length+beg:train_length+beg + 1]
    pred_y = pred_y.reshape(-1, 1)
    print(pred_res.shape)
    print(pred_y.shape)
    print(pred_res)
    print(pred_y)
    plt.figure(3)
    plt.plot(pred_y, color='r', marker='o', label='right', linestyle='--')
    plt.plot(pred_res, color='b', marker='v', label='pred', linestyle='-.')
    #plt.plot(pred_y, 'r')
    #plt.plot(pred_res, 'b')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

elif exec_case is '6': #python stock_multis.py exec_case stride gap out_length ycol i_stock 
    last_predict = cell.predict(datax[-1:], None)
    last_y = datay[-1:]
    print('-------------- predic res -------------')
    if out_feat != 1:
        last_predict = last_predict[:,:,[close_icol]]
        last_y = last_y[:,:,[close_icol]]
    _last_predict = last_predict[:,:-1].reshape(-1, 1) #끝의 end token제거
    _last_y = last_y.reshape(-1, 1)
    plt.figure(3)
    plt.plot(_last_y, color='r', marker='o', label='right', linestyle='--')
    plt.plot(_last_predict, color='b', marker='v', label='pred', linestyle='-.')
    plt.xlabel('Time Period')
    plt.ylabel('Stock Price')
    plt.show()

elif exec_case is '7': #python stock_multis.py exec_case stride gap out_length ycol i_stock 
    print(rev_minmax(price, datax[-1:]))
    pred_res = cell.predict(datax[-1:], None)
    print('-------------- predic res -------------')
    if out_feat != 1:
        pred_res = pred_res[:,:,[close_icol]]
    pred_res = pred_res[:,:-1].reshape(-1, 1) #끝의 end token제거
    print(rev_minmax(price, pred_res))

"""
prev_length = 32, n_in_stock = 8
ISLE NAME: in_encode_isle_0
encord isle: (1, 256, 6) 256은 시퀀스 길이
multihead mask: 232 outer: 0
(232, 8, 6)
seq sz: 8 feat sz: 6 tsz: 1 stride: 8 kernel: 8 derive: 232 party: 1 reducing: 1 outsz: 1 각 스트라드 시퀀스(8)
kernel expand 2
kernel expand 2
kernel expand 2
kernel expand 2
enocde gate[batch, party, derive, latent]
(86, 1, 232, 32)
enocde ortho stride gate[batch, party, latent, cross_out]
(86, 1, 32, 1)
enocde gate[batch, party, cross_out(reducing), derive/latent]
(86, 1, 1, 32)
enocde gate[batch, outsz(party*reducing), derive/latent]
(86, 1, 32)
decode gate insz: 1 outsz: 8 [batch, outsz, derive(latent)]
(86, 8, 32)
out gate[batch, out_seq, out_feat]
(86, 8, 6)
calc loss[batch, out_seq, out_feat]
(86, 8, 6)
encord isle out: (1, 32, 32) 시퀀스 길이 8단위로 오토인코딩하는 것을 32번 스트라이드 반복하여 전체 길이 256를 분할 학습한 결과
ISLE NAME: in_encode_isle_1
encord isle: (1, 32, 32) 위에서 256시퀀스가 8길이가 1로 코드 압축되어 32시퀀스가 다음 인코딩 입력으로 
multihead mask: 232 outer: 0
(232, 8, 32)
seq sz: 8 feat sz: 32 tsz: 1 stride: 8 kernel: 8 derive: 232 party: 1 reducing: 1 outsz: 1 각 스트라드 시퀀스(8)
kernel expand 2
kernel expand 2
kernel expand 2
kernel expand 2
enocde gate[batch, party, derive, latent]
(11, 1, 232, 32)
enocde ortho stride gate[batch, party, latent, cross_out]
(11, 1, 32, 1)
enocde gate[batch, party, cross_out(reducing), derive/latent]
(11, 1, 1, 32)
enocde gate[batch, outsz(party*reducing), derive/latent]
(11, 1, 32)
decode gate insz: 1 outsz: 8 [batch, outsz, derive(latent)]
(11, 8, 32)
out gate[batch, out_seq, out_feat]
(11, 8, 32)
calc loss[batch, out_seq, out_feat]
(11, 8, 32)
encord isle out: (1, 4, 32) 시퀀스 길이 8단위로 오토인코딩하는 것을 4번 스트라이드 반복하여 전체 길이 32를 분할 학습한 결과
ISLE NAME: couple_isle
coax isle in: (1, 4, 32) 위에서 32시퀀스가 8길이가 1로 코드 압축되어 4시퀀스가 커플(듀얼인코더) 인코딩 입력으로
coax isle tar: (1, 16, 1) 타겟은 16시퀀스
multihead mask: 232 outer: 0
(232, 8, 1) 
위 4시퀀스가 pcode이고 bysz 16과 tsz 16이 같으므로 by gate에 타겟 쌍으로되는 입력은 없고 tsz 16이 커널 8단위 2개party로 분할
dual enocder bysz: 16 tsz: 16 p_sz: 4 stride: 8 kernel: 8 derive: 232 ni_part: 0 party: 2 reducing: 1 outsz: 16
kernel expand 2
kernel expand 2
kernel expand 2
kernel expand 2
dual enocder derive reduce 1 [batch, kernel, party, derive, latent]
(1, 8, 2, 232, 32)
dual enocder tstep derive reduce[batch, party, kernel, cross_out, latent]
(1, 2, 8, 1, 32)
dual enocder tstep reduce 0 conv: 8 lbound: 0.125 nt_part: 2 kernel: 8 psz: 4 party: 2 reducing: 1 [batch * nt_seq * kernel, (psz + party) * reducing, latent]
(16, 6, 32) pcode 사이즈 4와 타겟시퀀스 압축 길이 2를 더하여 6길이를 망사용 압축
multihead mask: 232 outer: 0
(232, 8, 32)
seq sz: 6 feat sz: 32 tsz: 1 stride: 8 kernel: 8 derive: 232 party: 1 reducing: 1 outsz: 1 입력 6길에 2가 제로패딩되어
kernel expand 2                                                                         8길이가 되어 1로 압축
kernel expand 2
kernel expand 2
kernel expand 2
enocde gate[batch, party, derive, latent]
(16, 1, 232, 32)
enocde ortho stride gate[batch, party, latent, cross_out]
(16, 1, 32, 1)
enocde gate[batch, party, cross_out(reducing), derive/latent]
(16, 1, 1, 32)
enocde gate[batch, outsz(party*reducing), derive/latent]
(16, 1, 32)
dual enocder out[batch, out seq, latent]
(1, 16, 32)
out gate[batch, out_seq, out_feat]
(1, 16, 1)
calc loss[batch, out_seq, out_feat]
(1, 16, 1)
"""