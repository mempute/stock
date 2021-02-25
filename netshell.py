
import numpy as np
import mempute as mp

#bsz은 실 사이즈보다 2개 크게, ysz은 1개 크개 설정
#psz은 0보다 크면 dynagen의 prev code input 사이즈, 0보다 작으면 generic의 prev code input 사이즈, 0이면 prev code input없음
class NetShell:
    def __init__(self, trc, psz = 64, bsz=64, ysz=32, latent_sz=32, indisc = 0, ydisc = 0, embedim = 0, feat = 1, prevfeat = 1, inplus = 0, pre_train = 0):
        self.trc = trc
        self.p_seq_len = psz
        self.b_seq_len = bsz
        self.y_seq_len = ysz
        self.x_seq_len = bsz - ysz
        self.target_len = ysz - 1 #-1은 go tokwn, end token 한개 제외
        self.feat_sz = feat #discret인 경우 1로 해야 함. by gate에 입력과 타겟을 함께 넣으므로 
        self.hidden_sz = latent_sz #입력과 타겟 피쳐 사이즈는 같아야 하고 사이즈를 틀리게 할려면
        self.embedim = embedim      #입력 파트를 in_gate에 psz 사이즈와 prevfeat피쳐 사이즈로 한다.
        self.indisc = indisc
        self.ydisc = ydisc
        self.inplus = inplus
        self.pre_train = pre_train
        self.bgate_bsize = 1

        if inplus: self.inplus_ysz = bsz #타겟에 입력값도 포함하여 목표 학습
        else: self.inplus_ysz = ysz
        if inplus and pre_train == 0: self.iaccu_prec = self.x_seq_len
        else: self.iaccu_prec = 0
        
        if inplus and self.x_seq_len == 0:
            print('inplus option error')
            exit()
        if pre_train and inplus == 0: 
            print('pre-train option error')
            exit()
        self.by = mp.flux(self.trc, [-1, bsz, feat], mp.variable, mp.tfloat)
        self.go = mp.flux(self.trc, [-1, 1, feat], mp.variable, mp.tfloat)
        self.end = mp.flux(self.trc, [-1, 1, feat], mp.variable, mp.tfloat)
        self.ytar = mp.flux(self.trc, [-1, self.inplus_ysz, feat], mp.variable, mp.tfloat)
        self.in_d = None
        self.tar_d = None
        
        mp.fill(self.go, 0.0)
        mp.fill(self.end, 0.0)
        
        if psz: 
            self.in_gate = mp.flux(self.trc, [-1, psz, prevfeat], mp.variable, mp.tfloat)
        else: 
            self.in_gate = None
        self.tar_gate = mp.flux(self.trc, [-1, self.inplus_ysz, feat], mp.variable, mp.tfloat)
        self.by_gate = mp.flux(self.trc, [-1, bsz, feat], mp.variable, mp.tfloat)
        #by gate에 입력과 타겟쌍을 구성할 경우 이 둘을 합한 discrete사이즈이고 입력파트가 없고 타겟만 있을
	    #경우 타겟만의 discrete사이즈이다. 타겟만의 사이즈일 경우 바이게이트 전체는 추론을 위한 사이즈이고 
	    #이때 입력은 in_gate에 이전문맥으로 구성한다. 다이나젠은 이전 문맥또는 입력 파트가 큰경우 이를 
	    #압축하기위해 사용한다. embedim은 듀얼 인코더의 임베딩 사이즈로 반드시 설정하고 latent_sz, 
	    #indiscret는 듀얼의 사이즐 다이나젠 생성자에서 명시한 것과 다르게 할 경우 설정한다.
        mp.setbygate(self.trc, self.by_gate, ysz, self.go, self.end, embedim)
        #pre train은 특정값을 예측하는것이 아니므로 오차측정없다. 따라서 정확도 측정은 목표값만이 대상이다.
        self.accu_input = mp.flux(self.trc, [-1, self.target_len, feat], mp.variable, mp.tfloat)
        self.accu_label = mp.flux(self.trc, [-1, self.target_len, feat], mp.variable, mp.tfloat)

        if psz > 0:
            self.im = mp.impulse(self.trc)
            self.dgen = mp.dynagen(mp.dgen_class, self.im, self.in_gate, self.tar_gate, latent_sz, 
                              indisc, ydisc, embedim)
            self.by_gate = mp.getbygate(self.dgen)
            mp.accuracy(self.im, self.accu_input, self.accu_label)
            self.cell = None
        else:
            self.im = None
            self.cell = mp.generic(self.in_gate, self.tar_gate, latent_sz, indisc, ydisc, embedim)
            mp.accuracy(self.cell, self.accu_input, self.accu_label)

    def recording(self):
        if self.im is not None: mp.recording(self.im)
        else: mp.save_weight(self.trc)

    def train(self, prev_ids, in_ids, tar_ids):#in_ids, tar_ids는 
        mp.fill(self.by, 0.0)#in_ids, tar_ids가 사이즈가 모자르면 제로패딩되게 리셋
        if in_ids is not None:
            if self.in_d is None: 
                self.in_d = mp.flux(self.trc, [in_ids.shape[0], in_ids.shape[1], self.feat_sz], mp.variable, mp.tfloat)
                mp.resizing4(self.go, self.in_d)
                mp.resizing4(self.end, self.in_d)
            mp.copya(self.in_d, in_ids)
            mp.howrite(self.by, self.in_d, 0, self.x_seq_len)#input, in_ids의 사이즈가 모자르면 위에서 제로 패딩됐음
        if self.tar_d is None: 
            self.tar_d = mp.flux(self.trc, [tar_ids.shape[0], tar_ids.shape[1], self.feat_sz], mp.variable, mp.tfloat)
        mp.copya(self.tar_d, tar_ids)
        #mp.printo(self.by, 2)
        #print('-------------------------')
        #mp.howrite(self.by, self.go, self.x_seq_len)#go token, 위에서 일괄 리셋했으므로 필요없음
        mp.howrite(self.by, self.tar_d, self.x_seq_len + 1, self.target_len)#target, +1은 go token, tar_ids가 더 크면 끝에 자동truc되어 이상없다.
        #by: input + <go token> + target
        #mp.printo(self.by, 2)
        if self.inplus:#타겟에 입력값도 포함하여 목표 학습
            if self.pre_train == 0:#ytar: input + target + <end token>
                mp.howrite(self.ytar, self.in_d, 0, self.x_seq_len)#input, in_ids의 사이즈가 모자르면 위에서 제로 패딩됐음
                mp.howrite(self.ytar, self.tar_d, self.x_seq_len, self.target_len)#target
                mp.howrite(self.ytar, self.end, self.x_seq_len + self.target_len)#end token
            #else: pre_train과정이면 오터인코딩(자기구조학습)으로서 
            #ytar는 input + <go token> + target 으로서 by값과 동일하므로 밑에서 by주입
        else:#ytar: target + <end token>
            mp.howrite(self.ytar, self.tar_d, 0, self.target_len)#target
            mp.howrite(self.ytar, self.end, self.target_len)#end token

        if self.p_seq_len != 0 and prev_ids is not None:
            mp.feeda(self.in_gate, prev_ids)#이전 문맥

        mp.feedf(self.by_gate, self.by) #by: input + <go token> + target

        if self.pre_train:
            mp.feedf(self.tar_gate, self.by) #by: input + <go token> + target
        else:# inplus이면 ytar는 input + target + <end token> 아니면 target + <end token>
            mp.feedf(self.tar_gate, self.ytar)

        if self.p_seq_len > 0: 
            total_loss, rv = mp.train(self.im)
        else:
            total_loss, rv = mp.train(self.cell)
        #print('loss: ', mp.eval(total_loss)[0])
        return total_loss, rv

    def predict(self, prev_ids, in_ids, tar_ids = None):
        if in_ids is not None:
            if self.in_d is None: 
                self.in_d = mp.flux(self.trc, [in_ids.shape[0], in_ids.shape[1], self.feat_sz], mp.variable, mp.tfloat)
                mp.resizing4(self.go, self.in_d)
                mp.resizing4(self.end, self.in_d)
            mp.copya(self.in_d, in_ids)
            mp.fill(self.by, 0.0) 
            mp.howrite(self.by, self.in_d, 0, self.x_seq_len)#input
            #mp.howrite(self.by, self.go, self.x_seq_len)#go token, 위에서 일괄 리셋했으므로 필요없음
            mp.feedf(self.by_gate, self.by)

        if self.p_seq_len != 0 and prev_ids is not None: 
            mp.feeda(self.in_gate, prev_ids)
            if self.bgate_bsize != prev_ids.shape[0]:#내부에서 추론과정에서 바이게이트를 사용하는데 
                self.bgate_bsize = prev_ids.shape[0] #입력과정에서는 사용되지않아 배치사이즈가 
                mp.resizing2(self.by_gate, self.bgate_bsize) #초기상태이므로 입력에 맞게 재설정한다.

        if tar_ids is not None:
            if self.tar_d is None: 
                self.tar_d = mp.flux(self.trc, [tar_ids.shape[0], tar_ids.shape[1], self.feat_sz], mp.variable, mp.tfloat)
            mp.copya(self.tar_d, tar_ids)
            mp.fill(self.ytar, 0.0)
            if self.inplus:#타겟에 입력값도 포함하여 오차값 평가
                mp.howrite(self.ytar, self.in_d, 0, self.x_seq_len)#input
                if self.pre_train:#ytar: input + <go token> + target
                    #mp.howrite(self.ytar, self.go, self.x_seq_len)#go token, 위에서 일괄 리셋했으므로 필요없음
                    mp.howrite(self.ytar, self.tar_d, self.x_seq_len + 1, self.target_len)#target, +1은 go token, tar_ids가 더 크면 끝에 자동truc되어 이상없다.
                else:#ytar: input + target + <end token>
                    mp.howrite(self.ytar, self.tar_d, self.x_seq_len, self.target_len)#target
                    #mp.howrite(self.ytar, self.end, self.x_seq_len + self.target_len)#end token
            else:#ytar: target + <end token>
                mp.howrite(self.ytar, self.tar_d, 0, self.target_len)#target
                #mp.howrite(self.ytar, self.end, self.target_len)#end token
            mp.feedf(self.tar_gate, self.ytar)

        if self.p_seq_len > 0: 
            if tar_ids is not None:
                self.y_pred, self.ploss = mp.predict(self.im, 1)
            else: 
                self.y_pred = mp.predict(self.im, 0)
        else: 
            if tar_ids is not None:
                self.y_pred, self.ploss = mp.predict(self.cell, 1)
            else: 
                self.y_pred = mp.predict(self.cell, 0)
        return mp.eval(self.y_pred)
    
    def predicts(self, prev_ids, input_ids, batch_size):
        y_preds = []
        if input_ids is not None: n = input_ids.shape[0]
        else: n = prev_ids.shape[0]
        i = 0
        if batch_size > n: batch_size = n
        while i + batch_size <= n:
            if prev_ids is not None:
                prev_batch = prev_ids[i:i + batch_size]
            else: prev_batch = None
            if input_ids is not None:
                in_batch = input_ids[i:i + batch_size]
            else: in_batch = None
            y_pred = self.predict(prev_batch, in_batch)
            #print(y_pred)
            y_preds.append(y_pred)
            i += batch_size
        if self.ydisc: y_preds = np.array(y_preds, dtype='i')
        else: y_preds = np.array(y_preds)
        return y_preds.reshape(-1, y_pred.shape[1], y_pred.shape[2]), i

    def __accuracy(self, predict_res, tar_ids):
        
        mp.copya(self.accu_input, predict_res)
        mp.copya(self.accu_label, tar_ids)
        if self.cell is None:
            error = mp.measure_accuracy(self.im)
        else:
            error = mp.measure_accuracy(self.cell)
        return error
    def _accuracy(self, predict_res, tar_ids): #끝의 end token제거
        _predict_res = predict_res[::, self.iaccu_prec:-1].copy()#critical 복사하지 않으면 사이즈 그대로
        return self.__accuracy(_predict_res, _tar_ids)

    def accuracy(self, prev_ids, in_ids, tar_ids, batch_size = 0):
        # 테스트용데이터로 rmse오차를 구한다, pre train은 특정값을 예측하는것이 아니므로 오차측정없다.
        #print("=======================")
        #print(in_ids.shape)
        if batch_size:
            predict_res, sz = self.predicts(prev_ids, in_ids, batch_size)
            predict_res = predict_res[0:sz, self.iaccu_prec:-1].copy()#critial 복사하지 않으면 사이즈가 그대로
            tar_ids = tar_ids[0:sz].copy()        #끝의 end token제거
        else:
            predict_res = self.predict(prev_ids, in_ids)
            if self.ydisc: predict_res = np.array(predict_res, dtype='i')
            predict_res = predict_res[::, self.iaccu_prec:-1].copy()#critical 복사하지 않으면 사이즈 그대로
        #print("+++++++++++++++++++++++++")     #끝의 end token제거                    
        #print(predict_res)
        #print("------------------")
        #print(tar_ids)
        error = self.__accuracy(predict_res, tar_ids)
       
        return mp.eval(error), in_ids, predict_res, tar_ids
"""
kmodel = NetShell(name='dual_dgen1d') #prev 64, input 32, target 32
#kmodel = NetShell(64, 64, 48, name='dual_dgen1d') #prev 64, input 16, target 48
#kmodel = NetShell(64, 64, 64, name='dual_dgen1d') #prev 64, input 0, target 64
#kmodel = NetShell(0, 64, 32, name='dual_dgen1d') #prev 0, input 32, target 32
kmodel.train(200000)
kmodel.predict()
"""