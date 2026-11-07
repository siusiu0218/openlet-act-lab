# OpenLET 仿真抓取与 ACT 实验

本项目记录从机器人仿真环境配置、脚本抓取基线，到 ACT（Action Chunking Transformer）训练与离线评估的实践。目标任务是在仿真中抓取物体、抬升并放入小推车，再完成拉杆触发与输送流程。

仓库按已完成的整理范围提供环境清单、仿真验收脚本和 ACT 工具。部分实验在原项目环境中完成，公开版本的可运行范围和验证状态如下。

## 项目进度

| 阶段 | 已公开内容 | 验证状态与剩余工作 |
| --- | --- | --- |
| 环境配置 | Slurm、Apptainer、ROS、MuJoCo版本记录与资源模板 | 原环境已运行仿真；公开部署包仍缺部分工具和二进制依赖 |
| 仿真抓取基线 | 动作验收、严格退出验收、脱敏结果 | 六轮历史日志回放与原结果一致；完整策略及启动链待整理 |
| ACT离线工具 | 误差与动作限值检查、检查点保存、验证缓存 | 来源实验有真实运行验证，公开版有CPU测试 |
| ACT训练与恢复 | 配置驱动入口、恢复契约、流程演示 | 实验性移植；真实ACT训练、跨进程恢复及周期评估集成待验收 |
| ROS闭环 | 当前保留接口相关辅助代码 | 动作准入未通过，真实闭环尚待验收 |

## 目录结构

```text
openlet-act-lab/
├── environment/       仿真环境与资源配置模板
├── simulation/        抓取动作和严格退出验收
├── openlet_act/       ACT指标、保存、缓存与训练模块
├── examples/          离线指标和保存恢复演示
├── configs/           ACT训练配置模板
├── tests/             CPU单元测试
├── docs/              使用说明、来源与脱敏结果
├── requirements.txt   CPU示例及测试依赖
└── README.md
```

## 1. 环境配置与仿真抓取

原仿真环境采用Rocky Linux宿主、Slurm资源调度和Apptainer容器，容器内为Ubuntu20.04与ROS Noetic。C++和Python MuJoCo版本分别为3.0.1和3.2.3。环境模板中的路径与镜像SHA需要在实际部署时填写。

脚本抓取基线依次完成场景初始化、三个圆柱物体抓取、至少8cm实际抬升、目标箱中心入箱检查，以及拉杆与输送流程。动作结果与退出状态分别验收。

2026-10-25的来源实验中，seed28两轮动作及严格退出均通过。seed29、30、31各两轮，动作6/6通过、严格退出5/6通过；最后一轮nodelet退出134。六轮实际抬升约10.05–11.70cm。该小样本结果适用于对应脚本及仿真版本。

| 入口 | 用途 |
| --- | --- |
| [环境模板](environment/simulation.example.json) | 版本、资源和待填写部署参数 |
| [仿真说明](docs/simulation.md) | 验收命令、完整启动链依赖和已知缺项 |
| [动作验收](simulation/check_task.py) | 抓取阶段、抬升、入箱及输送检查 |
| [退出验收](simulation/check_exit.py) | 进程退出、图像/状态、库映射和资源释放标记检查 |
| [脱敏结果](docs/simulation-results.json) | 六轮仿真结果与各物体抬升数值 |

验收脚本读取对应版本的运行日志，并向输入目录写入结果。请按仿真说明对日志副本执行。完整仿真运行需要补齐说明中列出的依赖。

## 2. 本地运行 ACT 示例

使用Python3.10+，在仓库根目录执行。以下示例与测试仅需NumPy，使用合成数据。

```sh
python -m pip install -r requirements.txt
python -m examples.offline_metrics
python -m examples.training_resume
python -m unittest discover -v
```

离线指标示例输出逐轴误差和越界统计。保存恢复示例使用标量回归，验证连续12步与保存后恢复5+7步结果一致。示例限值只用于测试。

当前24项CPU测试通过，覆盖指标计算、padding、异常输入、检查点失败保护、缓存损坏、训练预算和恢复契约。仿真验收另外在六轮原始日志副本上完成回放。

## 3. ACT训练与恢复

```sh
python -m openlet_act.long_act --config configs/training.example.json --check-config
```

这条命令检查配置结构。实际训练需要兼容的PyTorch/LeRobot环境、本地数据、初始模型及完整输入SHA清单。训练入口会保存模型、处理器、AdamW、随机数状态及采样游标。恢复要求输入身份和阶段配置匹配。

来源实验于2026-11-06完成4000→10000步训练，固定350输入评估及独立复算通过。10000步normalized L1为0.164243，首步超过候选位置限值0.01rad的输入为224/350，动作准入失败。350输入已用于模型选择，独立最终测试仍待完成。

公开训练入口经过移植，真实ACT运行尚待目标环境验收。固定350输入评估当前作为独立函数提供，周期回调仍待接入。配置、命令、超时与恢复限制见[训练说明](docs/training.md)。

## ACT模块与数据约定

| 模块 | 接口与职责 |
| --- | --- |
| `openlet_act.offline_metrics` | `metrics` / `report`计算最多35输入的逐轴MAE、RMSE、P95、越界与动作跳变 |
| `openlet_act.long_offline_metrics` | 显式设置`max_inputs=350`扩展输入上限 |
| `openlet_act.inference_contract` | 输入、限值及CPU协议验证 |
| `openlet_act.checkpoint_io` | `save_checkpoint`写入暂存目录、校验SHA、发布完成回执并更新latest |
| `openlet_act.validation_cache` | `ValidationCache`缓存固定350个输入，逐文件校验SHA |

评估输入包括float32的state `(N,18)`、target/normalized/prediction `(N,100,21)`及bool的is_pad `(N,100)`。state对应右臂7轴和手部11轴，action另含底盘3维。records包含episode、frame、length，评估episode限定24–30。

prediction和target使用模型后处理后的数据集单位，手部乘1.57转换为rad，底盘单位为m/s、m/s、rad/s。位置与变化限值同样使用数据集单位。目标误差排除padding，完整预测块另行检查越界。normalized字段保留作证据，基础指标函数不使用它计算归一化损失。

检查点工具适用于单写入者；`existing_root=True`以完成回执判断逐文件发布是否完整，断电持久性和任意嵌套目录校验尚无保证。缓存单文件上限10MiB、默认总量3GiB；真实使用需PyTorch，并仅加载可信作业目录中的缓存。

## 来源与许可

ACT工具源自OpenLET管理仓库的实验代码，仿真验收工具来自对应基线实验。来源提交、原始或提取版本SHA记录见[来源清单](docs/source_manifest.json)。目录重组及公开适配后的文件可能与原始SHA不同。

官方机器人源码、LeRobot及PyTorch依赖的许可由各自项目规定。完整仿真源码的再分发范围仍待核对。本仓库尚未选择开源许可证，当前未授予额外再分发许可。

个人配置、凭据文件、数据、模型和运行目录由`.gitignore`排除。配置模板使用占位路径，公开提交使用GitHub隐私邮箱。
