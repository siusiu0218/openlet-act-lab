# 仿真环境与抓取基线

本阶段验证机器人在仿真中启动、抓取三个圆柱物体、抬升、放入目标箱/小推车，并操作拉杆完成输送。策略为脚本轨迹基线。

## 已整理代码

`simulation/check_task.py` 是实际实验使用的动作验收脚本。输入目录含policy.log、process-exit-code.txt、task1_grasp_diagnostics.jsonl、task_result.txt。检查三物体阶段记录、各物体实际抬升至少8cm、官方中心入箱条件、拉杆与输送完成标记。输出acceptance.json，返回0代表通过，1代表失败。目标箱检查仅确认官方中心入箱条件。

```sh
OPENLET_TEST_SEED=29 python simulation/check_task.py /path/to/copied-run/task1 full
```

`simulation/check_exit.py` 是实际实验的严格退出验收脚本。输入目录含model-entry.log、egl-selection-passed.txt、observations/summary.json和runtime-maps.json。检查nodelet/MPC退出码、EGL及物理线程释放、图像/状态和库映射验收标志，输出exit-check.json。以下环境变量须来自真实运行回执。

```sh
OPENLET_OUTPUT=/path/to/copied-run OPENLET_TEST_SEED=29 OPENLET_INITIAL_RC=0 OPENLET_FORCED=0 OPENLET_ENTRY_RC=130 python simulation/check_exit.py
```

两个脚本会在输入目录写结果，请对日志副本执行。缺失数据应失败；畸形JSON可能抛异常。它们依赖对应版本的日志契约，使用前核对版本。

## 环境与启动依赖

环境记录见 `environment/simulation.example.json`。这是待填写路径及镜像SHA的清单。历史宿主为Rocky Linux，容器为Ubuntu20.04/ROS Noetic，经Slurm分配后运行Apptainer；C++/Python MuJoCo分别为3.0.1/3.2.3。

历史调用链为full-new.sh → exec.sh → full-new-inner.sh → model_entry.py和task1.py。除这些文件，还依赖gpu-safety.sh、network-health.py、output-watchdog.py、verify-release-selection.sh、clock-observer.py、process_perf_observer.py、select-scene.py、capture-maps.py、camera_observer.py，以及匹配的Drake所有权库、控制器和EGL组件。部分依赖在外部bundle，当前公开目录尚未组成完整可运行仿真环境。

上游基准62de744c，已验证适配版本7387173e。抓取主程序位于独立源码仓库的src/data_challenge_simulator/examples/task1.py，物体状态接口位于utils/object_pos.py，轨迹控制位于utils/trajectory_controller.py。7387173e的改动将lift/transfer每点0.02秒保持时间绑定物理仿真时间；时钟缺失、回退或停滞会失败，每步墙钟等待最多2秒。

当前未找到该仿真子目录明确的许可证文件，完整官方策略及第三方源码的再分发范围待确认。此批整理保留源码身份和路径，完整启动部署包需要补齐依赖、逐项审阅再分发范围及目标环境验证。

