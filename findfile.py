import os
from typing import List, Optional

def find_result_files(
    directory: str,
    algorithms: Optional[List[str]] = None,
    other: Optional[List[str]] = None,
) -> List[str]:
    """
    查找符合要求的文件名，并返回完整路径
    
    Args:
        directory: 要搜索的目录路径（如 'output/0'）
        algorithms: 算法名称列表（如 ['FedMR', 'FedAvg']）
        other: 其他需要包含的字符串列表（如 ['test_loss', 'T_0.5']）
    
    Returns:
        符合条件的完整文件路径列表（如 ['output/0/FedMR_..._T_0.5.txt', ...]）
    """
    # 检查目录是否存在且不为空
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"'{directory}' is not a directory")
    
    if not os.listdir(directory):
        return []
    
    # 初始化参数
    algorithms = algorithms or []
    other = other or []
    
    matched_files = []
    
    for filename in os.listdir(directory):
        # 检查算法条件
        if algorithms:
            algorithm_match = any(alg in filename for alg in algorithms)
            if not algorithm_match:
                continue
        
        # 检查其他条件
        if other:
            other_match = all(item in filename for item in other)
            if not other_match:
                continue
        
        # 拼接完整路径
        full_path = os.path.join(directory, filename)
        matched_files.append(full_path)
    
    return matched_files

