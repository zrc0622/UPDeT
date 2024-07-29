# 训练
```bash
# UPDeT (baseline)
python3 src/main.py --config=vdn --env-config=sc2 with env_args.map_name=5m_vs_6m

# Skill UPDeT
python3 src/skill_main.py --config=vdn --env-config=sc2 with env_args.map_name=5m_vs_6m
```