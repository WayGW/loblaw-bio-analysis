# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# Paths
conda_env = r'C:\Users\waygw\miniconda3\envs\loblaw-bio'
xgboost_dll = r'C:\Users\waygw\miniconda3\envs\loblaw-bio\Library\mingw-w64\bin\xgboost.dll'
xgboost_pkg = r'C:\Users\waygw\miniconda3\envs\loblaw-bio\Lib\site-packages\xgboost'

print(f"Conda environment: {conda_env}")
print(f"XGBoost DLL: {xgboost_dll}")
print(f"XGBoost package: {xgboost_pkg}")

# Check files exist
if os.path.exists(xgboost_dll):
    print("✓ XGBoost DLL found!")
else:
    print("✗ XGBoost DLL NOT found!")

if os.path.exists(xgboost_pkg):
    print("✓ XGBoost package found!")
else:
    print("✗ XGBoost package NOT found!")

# Binaries - Copy XGBoost DLL to multiple locations
binaries = []

if os.path.exists(xgboost_dll):
    binaries.append((xgboost_dll, 'xgboost/lib'))
    binaries.append((xgboost_dll, 'lib'))
    binaries.append((xgboost_dll, 'bin'))
    binaries.append((xgboost_dll, 'Library/mingw-w64/bin'))
    binaries.append((xgboost_dll, '.'))
    print(f"  Copying xgboost.dll to 5 locations")

# Data files - Include entire xgboost package
datas = [
    ('config', 'config'),
    ('src', 'src'),
    ('data/raw/cell-count.csv', 'data/raw'),
]

# Add entire xgboost package (includes VERSION and other metadata)
if os.path.exists(xgboost_pkg):
    datas.append((xgboost_pkg, 'xgboost'))
    print(f"  Including xgboost package directory")

# Hidden imports
hiddenimports = [
    'PIL',
    'PIL._tkinter_finder',
    'pandas',
    'numpy',
    'scipy',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    'scipy.sparse._sparsetools',
    # scikit-learn
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.ensemble._base',
    'sklearn.tree',
    'sklearn.tree._tree',
    'sklearn.tree._utils',
    'sklearn.utils',
    'sklearn.utils._cython_blas',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'sklearn.preprocessing',
    'sklearn.preprocessing._data',
    'sklearn.model_selection',
    'sklearn.model_selection._split',
    'sklearn.model_selection._validation',
    'sklearn.metrics',
    'sklearn.metrics._scorer',
    'sklearn.metrics._classification',
    'sklearn.metrics._ranking',
    # XGBoost
    'xgboost',
    'xgboost.core',
    'xgboost.sklearn',
    'xgboost.training',
    'xgboost.compat',
    'xgboost.callback',
    'xgboost.libpath',
]

# Exclude problematic modules
excludes = [
    'xgboost.testing',
    'xgboost.spark',
    'xgboost.dask',
    'hypothesis',
    'pytest',
    'IPython',
    'jupyter',
    'notebook',
    'streamlit',
]

a = Analysis(
    ['gui_app.py', 'simple_viewer.py'],
    pathex=[conda_env],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LoblawBio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LoblawBio',
)