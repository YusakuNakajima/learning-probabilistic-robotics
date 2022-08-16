# %% [markdown]
# ## 5章：パーティクルフィルタによる自己位置推定(後半)
# 

# %% [markdown]
# - ここまでで、ロボットの姿勢のバラつきを実装してきた
# - ただ、これだとただパーティクルが拡散するだけでロボットの姿 勢はだんだんと分からなくなる
# - よって、センサの情報からこれを補正する必要がある
# 
# ### センサ値によるパーティクルの姿勢の評価
# - まずは単純な例
#   - あるセンサ値が得られたときに、二つのパーティクルのどっちが真値にふさわしいか？
#     - 当然、ロボットに近い方がふさわしい
#   - これは観測モデルの比で表せる
#     - 例：$p_j(z_j|x^{(i)})=0.02,p_j(z_j|x^{(k)})=0.01$とする
#     - $x^(i)$の方が$x^(k)$より2倍尤もらしいといえる
#       - この値は確率ではないが、比較は可能
#       - 比：尤度比、数値：尤度
#       - $p_j(z_j|x^{(i)})$は自明ではないので、後ほど実験から求める
# 
# **尤度関数**
#   - $p_j(z_j|x^{(i)})$について、本来zを取得してxを変数とするはずが逆になっている
#     - xを変数とみなした尤度関数で表現
#     - $L_j(x|z_j)=\eta _j(z_j|x)$
#       - $L_j()$というのが尤度
#       - 比を使うだけなので、$\eta$は正であれば何でもよい
# 
# **ベイズの定理からの導出**
# - $b_t(x_t^{(i)})=\hat{b_t}(x_t^{(i)}|z_{j,t})$
# - $= \eta p_j(z_{j,t}|x_t^{i})\hat{b_t}(x_t^{(i)})$
#   - 事前信念に対して、パーティクルの姿勢を決めた上でのセンサの値による確率密度を用いて信念の更新を行う
# - $= \eta L_j(x_t^{i}|z_{j,t})\hat{b_t}(x_t^{(i)})$
#   - 当然、パーティクルの姿勢とセンサ値の条件付確率についてひっくり返すこともできる
#   - なおここの事前分布$\hat{b_t}(x_t^{(i)})$は1になる、これまで全てのパーティクルの主にを同じに考えていたため
# 
# **パーティクルの重み**
# - 尤度は重みとしてパーティクルの分布に反映する
#   - 今まではロボットの分身として$x_t-{(i)}$で考えていたが重み$w_t-{(i)}$も含めるとして再定義
# - パーティクルの再定義
#   - $\xi _t^{(i)}=(x_t-{(i)},w_t-{(i)})$
#   - $\Sigma _{i=0}^{N-1} w^{(i)}=1$
#     - 重みは足したら1
# - 信念分布を次のように定義
#   - $P(x_t^* \in X)=\int _{x \in X}b_t(x)dx \approx \sigma_{i=0}^{N-1}w_i^{(i)} \delta(x_t^{(j)} \in X)$
#     - Xに真の姿勢が含まれる確率を、パーティクルの重み付き和で近似
# - 重みの計算(後ほど正規化)
#   - $w_t^{(i)}=L_j(x_t^{(i)}|z_{j,t})\hat w_t^{(i)}$
#     - 重みは尤度関数でアップデートすればよい
# 
# **尤度関数の設計**
# - $L_{j}(x \mid z_{j})=\mathcal{N}[z=z_{j} \mid h_{j}(x), Q_{j}(x)]$
# - $Q_{j}(x)=(\begin{array}{cc}{[\ell_{j}(x) \sigma_{\ell}]^{2}} & 0 \\ 0 & \sigma_{\varphi}^{2}\end{array})$
# - $\ell_{j}(x)$:$x$ とランドマーク $\mathrm{m}_{j}$ の距離
# - $h_{j}$ : 観測関数  
# - 用は、距離方向の分散と角度方向の分散だけど考えて、共分散を0とする
#   - 距離方向だけ$\ell_{j}(x)$かかっているのは、距離に比例して分散が大きくなることを表現するため
# - 実は、4章の誤差モデルと同じにしている
#   - 細かく決めると大変なので、とりあえずこれでやる
# - ロボットの時と同様、センサの統計をとってこの尤度関数の分散のパラメータを決める
#   - 教科書だと$\sigma_{\ell}=0.14[m/m]$,$\sigma_{\varphi}=0.05[rad]$が求まった
# 
# ## リサンプリング
# - ※本だけだと分かりにくいので、動画参照した方がいい
#   - アルゴリズムは意外と単純
# - このままだと、尤もらしいパーティクルに重みが集中し、最終的に1つのパーティクルだけに重みがついてパーティクルの意味がなくなる
# - 重みの大きすぎるパーティクルを複数に分割する
#   - 同時に重みの小さいパーティクルを消して全体のパーティクル数を一定に保つ
# - これをリサンプリングで実装
#   - サンプリングした標本から再びサンプリングをすること
# 
# **単純なリサンプリング**
# - 重みに応じて選ばれる確率を決めて、パーティクルを一つ選ぶ
# - 選んだパーティクルのコピーを作成し、重みは1/Nとする
# - これをN回繰り返してN個のコピー集合を新たなパーティクルの集合とする
# - 問題点
#   - 計算量が大きい
#     - アルゴリズムとしては、まず重みの積み上げリストを作り、合計値から一つの値を乱数で選ぶ
#     - 選ばれた乱数に相当するパーティクルをコピーする
#       - このとき、「選ばれた乱数に相当するパーティクル」は二分探索で選ばれるので計算量がO(logN)
#       - これをパーティクルN個で毎回探索するので最終的な計算量はO(NlogNになる)、できればO(N)にしたい
#   - サンプリングバイアス
#     - 同じ重みがあった時に同じ割合で選ばれるとは限らない
#       - 例えば5個の同じ重みのパーティクルから5個選ぶ問題を考える
#         - これは重複ありの組み合わせ問題と考えられるので、計算すると126通りになる
#           - 参考：http://www.geisya.or.jp/~mwm48961/kou2/s1combi5.htm
#         - 5個のパーティクルを1個ずつ選ぶ確率は1/126になるので、パーティクルの分布が崩れるためよくない
#         - つまりサンプリングバイアスがある
#   
# **系統リサンプリング**
# - 裏話として、確率ロボティクスの執筆時に用語が分からず「等間隔サンプリング」と訳したとか・・・
# - アルゴリズム
#   - まず重みの積み上げリストを作り、0~1/Nの間から一つの値を乱数で選ぶ(初期位置だけ乱数で選ぶ)
#   - その後、N/1ずつリストを進みながら合計でN個のパーティクルを選ぶ
#     - 単純なリサンプリングだと毎回乱数で選んでいたのに対して、こっちは1/Nづつ等間隔で選んでいく
#   - コピーの重みをNで割って新たなパーティクルとする
# - 問題点の解決
#   - 計算量がすくない
#     - 1回がO(1)でN回だとO(N)
#   - サンプリングバイアス
#     - 毎回乱数ではなくなったので、単純なサンプリングで生じたバイアス問題も解決
# - ネットで調べると、単純なサンプリングよりもランダム性が低いのでリサンプリングとして微妙とするものもある
# 
# ## 出力の実装
# - パーティクルフィルタの結果はパーティクルの分布そのものであるが、12章のPOMDP以外は分布ではなく結果を1つの値に集約して扱う(何らかのアルゴリズムで決めた値を最尤値とする)
#   - 本書執筆時点の2020年でも、分布そのものを扱えるアルゴリズムはあまりないらしい
# - いくつか考えられるが、ここでは最も重みの大きいパーティクルの姿勢(つまりパーティクルのモード)を返す
#   - 他には期待値をとる(平均をとる)ものもあるが、分布が2つ以上に別れた場合、その間のパーティクルの推定範囲外の座標を返すことがあるので良くない
#   - 用は分布のどこが尤もらしいかを決定する問題であるが、パーティクルフィルタの場合は分布が常に変化するので平均とモードのどっちが良いと一概に決めることができない
#     - 強いて言えば分布全体を使うのが適切

# %%
# conect to drive on colab
# from google.colab import drive
# drive.mount("/content/drive")
# dir_path="./drive/MyDrive/Colab Notebooks/ProbabilisticRobotics/"
dir_path="./"

import sys
sys.path.append(dir_path)
from robot import *

from scipy.stats import multivariate_normal
import numpy as np
pi=np.pi

import random
import copy

# %%
class Particle:
    def __init__(self,init_pose,weight):
        self.pose=init_pose
        self.weight=weight
        
    def motion_update(self,nu,omega,time,noise_rate_pdf):
        ns=noise_rate_pdf.rvs()
        noised_nu=nu+ns[0]*np.sqrt(nu/time)+ns[1]*np.sqrt(omega/time)
        noised_omega=omega+ns[2]*np.sqrt(nu/time)+ns[3]*np.sqrt(omega/time)
        self.pose=IdealRobot.state_transition(noised_nu,noised_omega,time,self.pose)
    
    def observation_update(self,observation,envmap,distance_dev_rate,direction_dev):
        for d in observation:
            obs_pos=d[0]
            obs_id=d[1]

            # calculate distance and direction of landmark from location of particles and map
            pos_on_map=envmap.landmarks[obs_id].pos
            particle_suggest_pos=IdealCamera.observation_function(self.pose,pos_on_map)

            # calclate likelihood 
            distance_dev=distance_dev_rate*particle_suggest_pos[0]
            cov=np.diag(np.array([distance_dev**2,direction_dev**2]))
            self.weight*=multivariate_normal(mean=particle_suggest_pos,cov=cov).pdf(obs_pos)
    

# %%

class Mcl:
    def __init__(self,envmap,init_pose,num,motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
    distance_dev_rate=0.14,direction_dev=0.05):
        self.particles=[Particle(init_pose,1/num) for i in range(num)]
        self.map=envmap
        self.distance_dev_rate=distance_dev_rate
        self.direction_dev=direction_dev

        # 4次元のガウス分布のオブジェクトを作成
        # diagは対角行列の生成
        v=motion_noise_stds
        c=np.diag([v["nn"]**2,v["no"]**2,v["on"]**2,v["oo"]**2])
        self.motion_noise_rate_pdf=multivariate_normal(cov=c)
        self.ml=self.particles[0]
        self.pose=self.ml.pose

    def set_ml(self):
        i=np.argmax([p.weight for p in self.particles])
        self.ml=self.particles[i]
        self.pose=self.ml.pose

    def motion_update(self,nu,omega,time):
            for p in self.particles:
                p.motion_update(nu,omega,time,self.motion_noise_rate_pdf)

    def observation_update(self,observation):
        for p in self.particles:
                p.observation_update(observation,self.map,self.distance_dev_rate,self.direction_dev)
        self.set_ml()
        self.systematic_resampring()
        # self.simple_resampring()
    
    def simple_resampring(self):
        ws=[e.weight for e in self.particles]
        if sum(ws)>1e-100: ws=[e+1e-100 for e in ws]#重みの和が0でエラーにならないように調整
        ps=random.choices(self.particles,weights=ws,k=len(self.particles))#N個パーティクルを選ぶ
        self.particles=[copy.deepcopy(e) for e in ps]#選んだパーティクルを重みを均一に取り出す
        for p in self.particles:p.weight=1/len(self.particles)#重みの正規化
    
    def systematic_resampring(self):
        ws=np.cumsum([e.weight for e in self.particles])# 重みの累積積み上げリストを作成
        if sum(ws)>1e-100: ws=[e+1e-100 for e in ws]#重みの和が0でエラーにならないように調整
        
        step=ws[-1]/len(self.particles)
        r=np.random.uniform(0.0,step)#初期位置
        cur_pos=0
        ps=[]

        while(len(ps)<len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])# omit exception of over cur_pos
                r +=step
            else:cur_pos+=1

        
        ps=random.choices(self.particles,weights=ws,k=len(self.particles))#N個パーティクルを選ぶ
        self.particles=[copy.deepcopy(e) for e in ps]#選んだパーティクルを重みを均一に取り出す
        for p in self.particles:p.weight=1/len(self.particles)#重みの正規化

    
    def draw(self,ax,elems):
            xs=[p.pose[0] for p in self.particles]
            ys=[p.pose[1] for p in self.particles]
            vxs=[np.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
            vys=[np.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
            elems.append(ax.quiver(xs,ys,vxs,vys,\
                                    angles='xy', scale_units='xy', scale=1.5,color="blue",alpha=0.5))
            
    

# %%
class EstimationAgent(Agent):
    def __init__(self,time_interval,nu,omega,estimator):
        super().__init__(nu,omega)
        self.estimator=estimator
        self.time_interval=time_interval

        self.prev_nu=0.0
        self.prev_omega=0.0
             
    def decision(self,observation=None):
        self.estimator.motion_update(self.prev_nu,self.prev_omega,self.time_interval)
        self.prev_nu,self.prev_omega=self.nu,self.omega
        self.estimator.observation_update(observation)
        return self.nu,self.omega
   
    def draw(self,ax,elems):
        self.estimator.draw(ax,elems)
        x,y,t=self.estimator.pose
        s="({:.2f},{:.2f},{})".format(x,y,int(t*180/pi%360))
        elems.append(ax.text(x,y+0.1,s,fontsize=8))
   

# %%
# # パラメータ決めて描画テスト
# # motion_noise_stds={"nn":1,"no":2,"on":3,"oo":4}
# # motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2}
# from distutils.log import debug


# time_interval=0.1
# world = World(30, time_interval)

# m = Map()                                  
# m.append_landmark(Landmark(-4,2))
# m.append_landmark(Landmark(2,-3))
# m.append_landmark(Landmark(3,3))
# world.append(m)     

# init_pose=np.array([0,0,0]).T
# estimator=Mcl(m,init_pose,100)
# circling=EstimationAgent(time_interval,0.2,10/180*pi,estimator)
# r= Robot( init_pose, sensor=Camera(m), agent=circling,color="red")
# world.append(r)

# world.draw()



# %%



