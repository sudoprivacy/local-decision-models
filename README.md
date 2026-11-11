# Local decision models

本地决策模型训练与 AI 协作服务。首批案例是 [物理小问得分率预测](https://s.shareone.vip/s/difficulty-rubric-v2-yang) 和 [私人住宅 AI](https://s.shareone.vip/s/sudo-connect-control-private-ai)，共同遵循[本地训练架构 v0.2](https://s.shareone.vip/s/ben-di-jue-ce-mo-xing-xun-lian-yu-ai-xie-zuo-jia-gou)。

当前实现一个可执行的 **CPU 参考闭环**：注册数据 → 按完整时间/题组划分 → 独立 worker 做监督训练 → 检查点停止/恢复 → 结构化评测 → 主人批准影子版本 → 回放与回滚。训练使用随机初始化的微型 Qwen3 文本骨干和真实 PEFT LoRA，供验证代码链路。业务训练主线仍是 Qwen3.5-9B 决策层＋LoRA；目标模型、真实业务效果及设备验收见 [验收与后续顺序](docs/acceptance.md)。

## 本地运行

Python 3.13；当前锁定的是 CPU 环境，无需 API Key、远程模型调用或下载模型权重。

```powershell
uv sync --locked --group dev
uv run ldm init --root G:/local-decision-models-runtime/demo
uv run ldm smoke --root G:/local-decision-models-runtime/demo
uv run ldm serve --root G:/local-decision-models-runtime/demo
```

Linux 可将运行目录换成 `/srv/local-decision-models/demo`。服务默认仅绑定 `127.0.0.1:8793`。`init` 生成 owner/researcher/guest 三个本地凭据，保存在运行目录的 `credentials.json`；不要提交、截图或回显这些值。用对应 Bearer 凭据调用 `/docs` 所列 API。新作业只排队，单独执行一次 worker：

```powershell
uv run python -m ldm.worker --root G:/local-decision-models-runtime/demo --once
```

worker 有步骤、耗时、单线程和内存约束，并在 Windows 使用 BelowNormal 优先级。实际测试进程保持 Normal。参考 worker 一次只执行一个作业；不安装开机项、常驻清理器或计划任务。

## 两个首批案例

| 案例 | 当前可执行部分 | 进入真实业务验收前还需完成 |
|---|---|---|
| 物理 | 合成完整题目记录的训练/验证闭环；真实成绩 CSV 的本地接收核验与准备状态 | 全题背景、图示、答案/评分标准与 219 条真实成绩逐条对齐；固定模型 revision、目标设备与训练配方；真实留出评测 |
| 私人住宅 | 决策模型训练与影子版本；正常、歧义、越权、停机、确认和回滚的模拟网关回放 | 家庭数据授权、协议与设备接入、H200 实测、服务身份隔离、主人及伙伴现场验收 |

物理目标是学生群体的**平均得分率**，不等于模型解题成功率或置信度。候选得分率锚点的内部权重不构成学生完整得分分布。2026 年已被审阅，只能用于回顾性比较；独立或前瞻试卷仍待提供。旧的 15 维量表和历史精度不作为新模型效果。

可选登记本机已有私密 CSV，仅在本地读取并输出已核准的覆盖摘要：

```powershell
uv run ldm init --root G:/local-decision-models-runtime/physics-intake --physics-csv <本地observed_rates.csv>
uv run ldm readiness --root G:/local-decision-models-runtime/physics-intake
```

CSV 保留原位置，实际得分率不写入本仓库或 API 响应。仅有成绩而无完整题目时，代码返回 `training_ready=false`，并阻止将 CPU 流程测试配方用于真实数据。

## 协作接口与信息边界

`describe_dataset`、`prepare_dataset`、`create_split`、`run_experiment`、`get_diagnostics`、`compare_runs` 对应[模块说明](docs/architecture.md)。调用者仅提交已注册 ID、配方和预算；没有 SQL、文件下载、任意路径或 shell 接口。响应按字段白名单构造，错误不回显输入；小分组诊断抑制，权重/检查点/逐条预测/原文日志保留本地。

HTTP 接口约束已实现；当前单用户本地参考部署仍须进一步配置独立服务身份、文件权限和网络边界，才能形成部署隔离。现有 SudoCode/Nexus 继续拥有 agent、session 和 A2A 管理权；本仓库仅持有领域实验执行记录。

住宅模型只给建议。权限、范围、确认和关键设备限制由代码独立执行；内置网关只模拟，物理命令数始终为零。影子版本不提供生产发布入口。RLCD 式训练是后续独立研究阶段；当前监督损失不被标作 RLCD。

## 验证与资料

```powershell
uv run ruff check
uv run ruff format --check
uv run pytest
```

- [架构与接口职责](docs/architecture.md)
- [验收层级、实际限制与下一步](docs/acceptance.md)
- [原文版本与固定来源](docs/sources.json)

CI 只有一个离线 CPU 测试作业，不调用付费模型、不运行 live PTY、不下载预训练权重。真实数据、凭据和模型产物放在 Git 外部运行目录；私有仓库可见性不替代数据边界。
