# -*- mode: python ; coding: utf-8 -*-
# 这是 PyInstaller 的 spec 文件，用于定义如何将 Python 脚本打包为可执行文件。


 # Analysis 对象分析主脚本和依赖项，生成打包所需的元数据。
a = Analysis(
    ['jiemianyouxueya-onnxjiarubiaozhunhuaqi.py'],  # 主程序入口脚本
    pathex=[],      # 搜索路径，留空表示当前目录
    binaries=[],    # 需要包含的二进制文件
    datas=[],       # 需要包含的数据文件
    hiddenimports=[], # 隐式导入的模块
    hookspath=[],   # 自定义 hook 文件路径
    hooksconfig={}, # hook 配置
    runtime_hooks=[], # 运行时钩子
    excludes=[],    # 排除的模块
    noarchive=False, # 是否不使用单一归档包
    optimize=0,     # 优化级别
)
 # PYZ 对象将纯 Python 代码打包成一个归档文件（.pyz）。
pyz = PYZ(a.pure)

 # EXE 对象定义最终生成的可执行文件及其属性。
exe = EXE(
    pyz,                 # 打包的 Python 归档
    a.scripts,           # 主脚本
    a.binaries,          # 二进制文件
    a.datas,             # 数据文件
    [],                  # 额外的资源
    name='svco',         # 生成的可执行文件名
    debug=False,         # 是否生成调试版本
    bootloader_ignore_signals=False, # 是否忽略信号
    strip=False,         # 是否去除符号表
    upx=True,            # 是否使用 UPX 压缩
    upx_exclude=[],      # 不用 UPX 压缩的文件
    runtime_tmpdir=None, # 运行时临时目录
    console=True,        # 是否为控制台程序
    disable_windowed_traceback=False, # 是否禁用窗口化回溯
    argv_emulation=False, # 是否启用参数模拟（macOS）
    target_arch=None,    # 目标架构
    codesign_identity=None, # 代码签名（macOS）
    entitlements_file=None, # 权限文件（macOS）
)
