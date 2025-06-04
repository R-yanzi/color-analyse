import os
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import cv2

class HistoryManager:
    """标定历史记录管理器，负责保存和恢复标定历史"""
    
    def __init__(self, max_history=20):
        """
        初始化历史记录管理器
        
        Args:
            max_history: 最大历史记录数量
        """
        self.max_history = max_history
        self.history_root = Path("history")
        self.history_root.mkdir(exist_ok=True)
        self.current_image_path = None
        self.current_history_path = None
        self.history_entries = []
        
    def set_current_image(self, image_path):
        """设置当前图像路径，并初始化对应的历史记录目录"""
        if not image_path:
            return False
            
        self.current_image_path = Path(image_path)
        base_name = self.current_image_path.stem
        
        # 创建该图像的历史记录目录
        self.current_history_path = self.history_root / base_name
        self.current_history_path.mkdir(exist_ok=True)
        
        # 加载历史记录索引
        self._load_history_index()
        return True
        
    def save_snapshot(self, masks, description=None):
        """保存当前标定状态的快照"""
        if not self.current_history_path:
            return False
            
        # 创建时间戳
        timestamp = int(time.time())
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建快照ID
        snapshot_id = f"snapshot_{timestamp}"
        snapshot_dir = self.current_history_path / snapshot_id
        snapshot_dir.mkdir(exist_ok=True)
        
        # 准备掩码数据
        masks_data = []
        for mask_id, entry in masks.items():
            try:
                if entry is None or 'mask' not in entry:
                    continue
                    
                mask_array = entry["mask"]
                if mask_array is None or mask_array.size == 0:
                    continue
                    
                # 获取掩码属性
                color = entry.get("color", [0, 255, 0])
                visible = entry.get("visible", True)
                
                # 保存掩码图像
                mask_path = snapshot_dir / f"{mask_id}.png"
                cv2.imwrite(str(mask_path), mask_array.astype(np.uint8) * 255)
                
                # 记录掩码元数据
                mask_data = {
                    "id": mask_id,
                    "color": color,
                    "visible": visible,
                    "mask_file": mask_path.name
                }
                masks_data.append(mask_data)
            except Exception as e:
                print(f"[历史记录] 保存掩码 {mask_id} 时出错: {str(e)}")
                continue
                
        # 创建快照元数据
        snapshot_meta = {
            "id": snapshot_id,
            "timestamp": timestamp,
            "date": date_str,
            "description": description or "标定快照",
            "masks": masks_data
        }
        
        # 保存快照元数据
        meta_path = snapshot_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_meta, f, indent=2, ensure_ascii=False)
            
        # 更新历史记录索引
        self.history_entries.append(snapshot_meta)
        self._save_history_index()
        
        # 如果历史记录超过最大数量，删除最旧的记录
        self._cleanup_old_history()
        
        return snapshot_id
        
    def restore_snapshot(self, snapshot_id):
        """恢复指定的历史快照"""
        if not self.current_history_path:
            return None
            
        # 查找快照
        snapshot_dir = self.current_history_path / snapshot_id
        if not snapshot_dir.exists():
            print(f"[历史记录] 找不到快照: {snapshot_id}")
            return None
            
        # 加载快照元数据
        meta_path = snapshot_dir / "metadata.json"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                snapshot_meta = json.load(f)
        except Exception as e:
            print(f"[历史记录] 加载快照元数据失败: {str(e)}")
            return None
            
        # 恢复掩码
        restored_masks = {}
        for mask_data in snapshot_meta.get("masks", []):
            mask_id = mask_data.get("id")
            mask_file = mask_data.get("mask_file")
            color = mask_data.get("color", [0, 255, 0])
            visible = mask_data.get("visible", True)
            
            if not mask_id or not mask_file:
                continue
                
            # 加载掩码图像
            mask_path = snapshot_dir / mask_file
            if not mask_path.exists():
                print(f"[历史记录] 找不到掩码文件: {mask_path}")
                continue
                
            try:
                mask_array = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_array is None:
                    print(f"[历史记录] 无法读取掩码文件: {mask_path}")
                    continue
                    
                # 转换为布尔掩码
                mask_array = mask_array > 0
                
                # 创建掩码条目
                restored_masks[mask_id] = {
                    "mask": mask_array,
                    "color": tuple(color),
                    "visible": visible,
                    "editable": False
                }
            except Exception as e:
                print(f"[历史记录] 恢复掩码 {mask_id} 时出错: {str(e)}")
                continue
                
        return restored_masks
        
    def get_history_list(self):
        """获取历史记录列表"""
        return sorted(self.history_entries, key=lambda x: x.get("timestamp", 0), reverse=True)
        
    def _load_history_index(self):
        """加载历史记录索引"""
        self.history_entries = []
        if not self.current_history_path:
            return
            
        index_path = self.current_history_path / "index.json"
        if not index_path.exists():
            return
            
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                self.history_entries = json.load(f)
        except Exception as e:
            print(f"[历史记录] 加载历史记录索引失败: {str(e)}")
            
    def _save_history_index(self):
        """保存历史记录索引"""
        if not self.current_history_path:
            return
            
        index_path = self.current_history_path / "index.json"
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self.history_entries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[历史记录] 保存历史记录索引失败: {str(e)}")
            
    def _cleanup_old_history(self):
        """清理过旧的历史记录"""
        if len(self.history_entries) <= self.max_history:
            return
            
        # 按时间戳排序
        sorted_entries = sorted(self.history_entries, key=lambda x: x.get("timestamp", 0))
        
        # 删除最旧的记录
        to_remove = sorted_entries[:(len(sorted_entries) - self.max_history)]
        for entry in to_remove:
            snapshot_id = entry.get("id")
            if snapshot_id:
                snapshot_dir = self.current_history_path / snapshot_id
                try:
                    if snapshot_dir.exists():
                        shutil.rmtree(snapshot_dir)
                    self.history_entries.remove(entry)
                except Exception as e:
                    print(f"[历史记录] 删除旧快照 {snapshot_id} 时出错: {str(e)}")
                    
        # 更新索引
        self._save_history_index() 