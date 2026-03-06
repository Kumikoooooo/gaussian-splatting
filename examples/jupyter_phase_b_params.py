"""Jupyter 阶段 B（参数初始化）示例。

用法：把下面的代码按 cell 拆到 notebook 中执行，
最终得到 dataset / opt / pipe 三个对象：

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
"""

# ===== Cell 1: 导入 =====
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


# ===== Cell 2A: 方式一（推荐新手）继续使用 ArgumentParser =====
# 这和 train.py 最接近，只是把命令行参数改成 notebook 里直接传 list
parser = ArgumentParser(description="3DGS notebook params")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)

# train.py 里额外参数（这里给你常用的一组）
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=6009)
parser.add_argument("--debug_from", type=int, default=-1)
parser.add_argument("--detect_anomaly", action="store_true", default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--disable_viewer", action="store_true", default=True)  # notebook 建议先关 viewer
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default=None)

# 在 notebook 里等价于命令行：python train.py --source_path ... --model_path ...
args = parser.parse_args([
    "--source_path", "/path/to/your/data",      # TODO: 改成你的数据目录
    "--model_path", "./output/notebook_demo",   # TODO: 改成你的输出目录
    "--images", "images",                       # 数据里图片文件夹名
    "--iterations", "30000",                    # 可先改成 200 做冒烟测试
    "--eval",                                    # 可选：启用 eval 划分
])

# 与 train.py 一致：把最终迭代也加入 save 列表
args.save_iterations.append(args.iterations)

# 三个训练入口对象（你后续 training(...) 就要传它们）
dataset = lp.extract(args)
opt = op.extract(args)
pipe = pp.extract(args)

print("[Parser方式] 参数已就绪")
print("dataset.source_path =", dataset.source_path)
print("dataset.model_path  =", dataset.model_path)
print("opt.iterations      =", opt.iterations)
print("pipe.debug          =", pipe.debug)


# ===== Cell 2B: 方式二（Notebook 风格）手动构造 Namespace =====
# 不想写 parser 时可以用这个。优点是直观；缺点是容易漏参数。
# 注意：至少要覆盖你训练函数会访问到的字段。
manual_args = Namespace(
    # ModelParams 相关
    sh_degree=3,
    source_path="/path/to/your/data",            # TODO
    model_path="./output/notebook_demo_manual",  # TODO
    images="images",
    depths="",
    resolution=-1,
    white_background=False,
    train_test_exp=False,
    data_device="cuda",
    eval=False,

    # OptimizationParams 相关（先给一套默认值）
    iterations=30000,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30000,
    feature_lr=0.0025,
    opacity_lr=0.025,
    scaling_lr=0.005,
    rotation_lr=0.001,
    exposure_lr_init=0.01,
    exposure_lr_final=0.001,
    exposure_lr_delay_steps=0,
    exposure_lr_delay_mult=0.0,
    percent_dense=0.01,
    lambda_dssim=0.2,
    densification_interval=100,
    opacity_reset_interval=3000,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_grad_threshold=0.0002,
    depth_l1_weight_init=1.0,
    depth_l1_weight_final=0.01,
    random_background=False,
    optimizer_type="default",

    # PipelineParams 相关
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
    antialiasing=False,

    # train.py 附加字段
    ip="127.0.0.1",
    port=6009,
    debug_from=-1,
    detect_anomaly=False,
    test_iterations=[7000, 30000],
    save_iterations=[7000, 30000],
    quiet=False,
    disable_viewer=True,
    checkpoint_iterations=[],
    start_checkpoint=None,
)
manual_args.save_iterations.append(manual_args.iterations)

# 这里复用同一组 Param 类的 extract 逻辑
parser2 = ArgumentParser(description="manual args extractor")
lp2 = ModelParams(parser2)
op2 = OptimizationParams(parser2)
pp2 = PipelineParams(parser2)

dataset2 = lp2.extract(manual_args)
opt2 = op2.extract(manual_args)
pipe2 = pp2.extract(manual_args)

print("[Manual方式] 参数已就绪")
print("dataset2.source_path =", dataset2.source_path)
print("dataset2.model_path  =", dataset2.model_path)
print("opt2.iterations      =", opt2.iterations)
print("pipe2.debug          =", pipe2.debug)


# ===== Cell 3: 一个小检查（建议保留） =====
def quick_validate_params(dataset_obj, opt_obj, pipe_obj):
    assert isinstance(dataset_obj.source_path, str) and dataset_obj.source_path != "", "source_path 不能为空"
    assert isinstance(dataset_obj.model_path, str) and dataset_obj.model_path != "", "model_path 不能为空"
    assert opt_obj.iterations > 0, "iterations 必须 > 0"
    assert opt_obj.optimizer_type in ["default", "sparse_adam"], "optimizer_type 只能是 default/sparse_adam"
    assert hasattr(pipe_obj, "debug"), "pipe 缺少 debug 字段"
    print("参数检查通过 ✅")

# 任选一组检查
quick_validate_params(dataset, opt, pipe)
# quick_validate_params(dataset2, opt2, pipe2)
