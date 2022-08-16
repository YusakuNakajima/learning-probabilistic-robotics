# %% [markdown]
# ## 参考資料
# - [詳細確率ロボティクスのgit](https://github.com/ryuichiueda/LNPR_BOOK_CODES)
# ---
# ### 不確かさの実装
# - ロボットの不確かさ
#   - 雑音(ノイズ)
#     - 指数分布(expon)の確率でドロー
#     - ドローしたらposeにノイズを加える
#   - バイアス
#     - ガウス分布(norm)から初期化時にドロー
#     - ガウス分布の何%かをバイアスとして設定
#     - 速度と角速度にバイアスを加える
#   - スタック
#     - 指数分布(expon)の確率でドロー
#     - スタックが起きるまでの時間の期待値、起きてから抜けるまでの時間の期待値をそれぞれ計算
#     - スタック中は速度と角速度をゼロにする(現実のロボットはぶつかった衝撃で向きが変わったりするけど、ここでは簡略化)
#   - 誘拐
#     - 指数分布(expon)の確率でドロー
#     - ドローしたら座標を変更するが、変更後の座標は一様分布(uniform)で決定
# - センサーの不確かさ
#   - 雑音(ノイズ)
#     - 距離が長いほど大きくなるので距離に比例する標準偏差
#     - 角度はそこまでノイズのらないので一定の標準偏差
#   - バイアス
#     - ガウス分布(norm)から初期化時にドロー
#     - ガウス分布の何%かをバイアスとして設定
#     - 速度と角速度にバイアスを加える
#   - ファントム
#     - ないはずのランドマークをあると認識する
#     - 実装は一様分布でドローして引いた時にランドマーク追加
#   - 見落とし
#     - 文字通りの見落とし
#     - 実装は一様分布でひいて閾値の確率以下でセンサ値0にする
#   - オクルージョン
#     - みたいものが隠れる減少
#     - 実装では、見落としてないけど値が大きく違うという実装

# %%
# conect to drive on colab
# from google.colab import drive
# drive.mount("/content/drive")
# dir_path="./drive/MyDrive/Colab Notebooks/ProbabilisticRobotics/"
dir_path="./"

import sys
sys.path.append(dir_path)
from ideal_robot import *
from scipy.stats import expon,norm,uniform
import numpy as np
pi=np.pi

# %%
class Robot(IdealRobot):
    def __init__(self,pose,agent=None,sensor=None,color="black",
    noise_per_meter=5,noise_std=pi/60,
    bias_rate_stds=(0.1,0.1),
    expected_escape_time=1e-100,expected_stuck_time=1e100,
    expected_kidnap_time=1e100,kidnap_range_x=(-5,5),kidnap_range_y=(-5,5)
    ):
        super().__init__(pose,agent,sensor,color)
        # noise
        self.noise_pdf=expon(scale=1.0/(1e-100+noise_per_meter))
        self.distance_until_noise=self.noise_pdf.rvs()
        self.theta_noise=norm(scale=noise_std)
        # bias
        self.bias_rate_nu=norm.rvs(loc=1.0,scale=bias_rate_stds[0])
        self.bias_rate_omega=norm.rvs(loc=1.0,scale=bias_rate_stds[1])        
        # stuck
        self.escape_pdf=expon(scale=expected_escape_time)
        self.stuck_pdf=expon(scale=expected_stuck_time)
        self.time_until_escape=self.escape_pdf.rvs()
        self.time_until_stuck=self.stuck_pdf.rvs()
        self.is_stuck=False
        # kidnap
        self.kidnap_pdf=expon(scale=expected_kidnap_time)
        self.time_until_kidnap=self.kidnap_pdf.rvs()
        rx,ry=kidnap_range_x,kidnap_range_y
        self.kidnap_dist=uniform(loc=(rx[0],ry[0],0.0),scale=(rx[1]-rx[0],ry[1]-ry[0],pi*2))


    def noise(self,pose,nu,omega,time_interval):
        self.distance_until_noise-=abs(nu)*time_interval+self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()
        return pose

    def bias(self,nu,omega):
        return nu*self.bias_rate_nu,omega*self.bias_rate_omega

    def stuck(self,nu,omega,time_interval):
        if self.is_stuck:
            self.time_until_escape-=time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck=False
        else:
            self.time_until_stuck-=time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck=True
        return nu*(not self.is_stuck),omega*(not self.is_stuck)
    
    def kidnap(self,pose,time_interval):
        self.time_until_kidnap-=time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose

    def one_step(self,time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu,omega = self.agent.decision(obs)
        nu,omega=self.bias(nu,omega)
        nu,omega=self.stuck(nu,omega,time_interval)
        self.pose = self.state_transition(nu,omega,time_interval,self.pose)
        self.pose = self.noise(self.pose,nu,omega,time_interval)
        self.pose = self.kidnap(self.pose,time_interval)
    


# %%
class Camera(IdealCamera):
    def __init__(self,env_map,
        distance_range=(0.5,6.0),
        direction_range=(-pi/3,pi/3),
        distance_noise_rate=0.1,direction_noise=pi/90,
        distance_bias_rate_stddev=0.1,direction_bias_stddev=pi/90,
        phantom_prob=0.0, phantom_range_x=(-5.0,5.0), phantom_range_y=(-5.0,5.0),
        oversight_prob=0.1,
        occlusion_prob=0.0
        ) :
        super().__init__(env_map,distance_range,direction_range)

        self.distance_noise_rate=distance_noise_rate
        self.direction_noise=direction_noise  
        self.distance_bias_rate=norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias=norm.rvs(scale=direction_bias_stddev)

        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale=(rx[1]-rx[0], ry[1]-ry[0]))
        self.phantom_prob = phantom_prob

        self.oversight_prob=oversight_prob

        self.occlusion_prob=occlusion_prob

    def noise(self,relpos):
        ell = norm.rvs(loc=relpos[0],scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1],scale=self.direction_noise)
        return np.array([ell,phi]).T
    def bias(self,relpos):
        return relpos+np.array(relpos[0]*self.distance_bias_rate,self.direction_bias).T
    
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos)
        else:
            return relpos
    def oversight(self,relpos):
        if uniform.rvs()<self.oversight_prob:
            return None
        else:
            return relpos
    def occlusion(self,relpos):
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0]+uniform.rvs()*(self.distance_range[1]-relpos[0])
            phi = relpos[1]
            return np.array([ell,phi]).T
        else:
            return relpos
    def data(self,cam_pose):
        observed=[]
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose,lm.pos)
            z = self.phantom(cam_pose, z)
            z=self.occlusion(z)
            z=self.oversight(z)
            
            if self.visible(z):
                z=self.bias(z)
                z=self.noise(z)
                
                observed.append((z,lm.id))
        self.lastdata=observed
        return observed


# %%
if __name__ =='__main__':
    world = World(10, 0.1)


    m = Map()                                  
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)     


    circling=Agent(0.2,10/180*pi)
    r= Robot( np.array([0, 0, pi/6]).T, sensor=Camera(m,occlusion_prob=0.5), agent=circling)
    world.append(r)

    world.draw()

# %%



