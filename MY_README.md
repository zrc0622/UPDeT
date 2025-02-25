# 训练
```bash
# UPDeT (baseline)
python3 src/main.py --total-config=default_test --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m

# Phase UPDeT 1
python3 src/phase_main.py --total-config=phase_updet1 --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m

# Phase UPDeT 2
python3 src/phase_main.py --total-config=phase_updet2 --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m

python3 src/phase_main.py --total-config=phase_updet2_test --config=qmix --env-config=sc2 with env_args.map_name=8m_vs_9m

# Phase UPDeT 3
python3 src/phase_main.py --total-config=phase_updet3 --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
```

# 介绍
## 环境介绍
```
[DEBUG 18:44:30] absl ------------------------Obs Agent: 0------------------------
[DEBUG 18:44:30] absl Avail. actions [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

[DEBUG 18:44:30] absl Move feats [1. 1. 1. 1.]
移动mask，1表示该移动动作可执行

[DEBUG 18:44:30] absl Enemy feats [[ 1.          0.6628378   0.59868705 -0.28447807  1.        ]
 [ 1.          0.5838679   0.5834961  -0.02083333  1.        ]
 [ 1.          0.51339453  0.48209634 -0.17651367  0.8666667 ]
 [ 1.          0.59876114  0.5962185   0.05512153  0.73333335]
 [ 1.          0.5786748   0.5616862  -0.13918728  1.        ]
 [ 1.          0.59361845  0.552219   -0.21780056  1.        ]]
敌人状态，能否攻击、相对距离、相对x距离、相对y距离、血量值

[DEBUG 18:44:30] absl Ally feats [[ 1.          0.6099084  -0.60007054 -0.10910373  1.        ]
 [ 1.          0.23772596 -0.16175672 -0.1742079   1.        ]
 [ 1.          0.24877211 -0.15332031  0.19590929  0.6       ]
 [ 1.          0.24076377  0.01554362  0.2402615   0.73333335]]
[DEBUG 18:44:30] absl Own feats [0.6]
```
## 算法介绍
### divide q
`divide q`如果为`True`，那么`self mlp`和`interaction mlp`不共用
`divide q`如果为`False`，那么`self mlp`和`interaction mlp`共用，此时`interaction mlp`需要取均值

### input q phase
`input q phase`如果为`True`，那么在计算q值时引入阶段embedding（将阶段embedding与状态拼接进行计算）

### pqmix
`pqmix`在`qmix`的基础上引入了阶段embedding，多智能体的阶段embedding取mean后与state拼接参与mix

### pqmix_v2
`pqmix_v2`在`qmix`的基础上引入了阶段索引，多智能体的阶段索引concat后与state拼接参与mix

### 几种算法
#### TransT
#### DivideT
```
Phase UPDeT 2
QMIX
temperature 0.15
phase_num 6
divide_Q False
intput_Q_phase True
```
#### DivideP