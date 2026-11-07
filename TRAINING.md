# ACT 训练与恢复示例

## 验证状态

入口从原实验提取并改为本地配置驱动。原4000→10000步CUDA验收适用于原版本。移植入口已做CPU控制流程与恢复契约测试，真实PyTorch/LeRobot训练、处理器设备适配和跨进程模型恢复尚待目标环境验收。当前为实验性训练示例。

## CPU流程演示

```sh
python training_demo.py
python -m unittest discover -v
python long_act.py --config training.example.json --check-config
```

training_demo使用标量回归演示更新、检查点发布和游标恢复，比较连续12步与5+7步的精确结果。check-config只检查配置结构。

## ACT环境与配置

来源环境为Python3.12.12、PyTorch2.7.1+cu118、torchvision0.22.1+cu118、LeRobot提交a07f22e（项目适配版本），还使用NumPy、PyArrow、PyAV、safetensors和packaging。使用已验证兼容的LeRobot环境，通用最新版兼容性待验证。requirements.txt仅列CPU测试依赖。

准备本地LeRobot数据集，包括episode0–23训练段、10Hz、18维state、100×21 action窗口和与模型匹配的图像。初始checkpoint须含ACT权重、config.json、完整前后处理器配置和统计张量。处理器保存的设备须与device一致。

复制training.example.json到个人配置。填写dataset_root、checkpoint、独占的新output目录。相对路径以启动目录为基准。input_sha256须列出数据与模型目录内所有文件的路径和SHA256，入口按清单核验完整性。哈希会读取这些输入文件，按数据规模预留时间。

updates是阶段计划总更新数，save_every为保存间隔，budget_seconds为更新阶段软预算。固定AdamW lr=1e-5、weight_decay=1e-4、梯度裁剪10，workers=0。

```sh
python long_act.py --config training.local.json
python long_act.py --config resume.local.json --resume runs/stage-01/step-000500
```

首次从预训练ACT权重开始新优化器阶段。恢复配置仅需换新output；输入身份、seed、device、batch_size和updates须保持一致。恢复目录由此入口生成，包含AdamW、Python/NumPy/Torch/CUDA RNG、批次游标及完成回执。可信本地状态使用weights_only=False加载。旧集群检查点需另行转换验收。每阶段目标可配置，阶段内恢复要求目标一致。

输出含config.json、steps.jsonl、step-*、latest.json、summary.json。预算到期保存进度并返回2，正常完成返回0，异常停止。单次更新60秒、保存600秒硬超时退出124。软预算排除初始化和输入哈希；调度环境须设总时限并预留保存余量。

## 评估与边界

保存后校验回执和SHA。long_act_validation.evaluate保留固定350输入评估，参数为policy、pre/post、dataset、逐episode numeric数据、evaluation config、输出目录、步数和torch。提供episode24–30、训练统计一致的action_mean和limits。函数写入数组、记录、指标和receipt，恢复模型模式及RNG。update_selection记录误差和限值两种模型选择结果。

该评估函数尚未接入新CLI的周期回调。模型重载预测一致性、真实AdamW/RNG跨进程恢复和350输入评估仍须在目标环境验收。CPU测试覆盖预算、失败停止、采样顺序、恢复契约及指标行为。

学校集群运行前按项目规则核对Apptainer镜像、绑定、依赖和Slurm预算。新入口使用本地进程，调度器负责作业总时限及进程树清理。
