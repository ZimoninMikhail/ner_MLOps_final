"""
Модуль детекции дрейфа данных и автоматического переобучения
"""

import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from pathlib import Path

import numpy as np


class DriftDetector:
    """Детектор дрейфа данных для задачи NER"""
    
    def __init__(
        self,
        reference_stats_path: str,
        threshold_js_divergence: float = 0.15,
        threshold_new_tokens_ratio: float = 0.3,
        min_samples_for_detection: int = 50
    ):
        self.threshold_js = threshold_js_divergence
        self.threshold_tokens = threshold_new_tokens_ratio
        self.min_samples = min_samples_for_detection
        
        with open(reference_stats_path, 'r', encoding='utf-8') as f:
            self.reference_stats = json.load(f)
        
        self.reference_distribution = self.reference_stats.get('entity_type_distribution', {})
        self.reference_vocab = set(self.reference_stats.get('vocabulary', []))
    
    @staticmethod
    def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        kl = 0.0
        for key in p:
            if key in q and p[key] > 0 and q[key] > 0:
                kl += p[key] * np.log(p[key] / q[key])
        return kl
    
    @staticmethod
    def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        all_keys = set(p.keys()) | set(q.keys())
        m = {}
        for key in all_keys:
            m[key] = (p.get(key, 0) + q.get(key, 0)) / 2
        
        kl_pm = DriftDetector._kl_divergence(p, m)
        kl_qm = DriftDetector._kl_divergence(q, m)
        
        return (kl_pm + kl_qm) / 2
    
    def detect_drift(self, new_data_stats: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
        new_distribution = new_data_stats.get('entity_type_distribution', {})
        new_tokens = set(new_data_stats.get('vocabulary', []))
        new_count = new_data_stats.get('total_documents', 0)
        
        if new_count < self.min_samples:
            return False, {"error": f"Not enough data: {new_count} < {self.min_samples}"}
        
        js_div = self._js_divergence(self.reference_distribution, new_distribution)
        
        if self.reference_vocab:
            new_tokens_ratio = len(new_tokens - self.reference_vocab) / len(new_tokens) if new_tokens else 0
        else:
            new_tokens_ratio = 0
        
        drift_metrics = {
            "js_divergence": js_div,
            "new_tokens_ratio": new_tokens_ratio,
            "reference_samples": self.reference_stats.get('total_documents', 0),
            "new_samples": new_count,
            "has_drift": (js_div > self.threshold_js) or (new_tokens_ratio > self.threshold_tokens)
        }
        
        return drift_metrics["has_drift"], drift_metrics


def compute_data_statistics(processed_data_dir: str, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Вычисление статистики для детекции дрейфа"""
    stats = {
        "total_documents": 0,
        "total_entities": 0,
        "entity_type_distribution": {},
        "vocabulary": [],
        "avg_entities_per_doc": 0
    }
    
    json_files = []
    for root, dirs, files in os.walk(processed_data_dir):
        for file in files:
            if file.startswith("doc_") and file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    if sample_size:
        json_files = json_files[:sample_size]
    
    stats["total_documents"] = len(json_files)
    
    all_tokens = []
    entity_counter = Counter()
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        
        text = doc.get('text', '')
        tokens = text.split()
        all_tokens.extend(tokens)
        
        # Используем parsed_entities, а не entities
        entities = doc.get('parsed_entities', [])
        stats["total_entities"] += len(entities)
        
        for ent in entities:
            if isinstance(ent, dict):
                ent_type = ent.get('type', 'UNKNOWN')
            elif isinstance(ent, str):
                ent_type = ent
            else:
                ent_type = 'UNKNOWN'
            entity_counter[ent_type] += 1
    
    total_ents = stats["total_entities"]
    if total_ents > 0:
        stats["entity_type_distribution"] = {k: v / total_ents for k, v in entity_counter.items()}
    
    stats["vocabulary"] = list(set(all_tokens))
    if stats["total_documents"] > 0:
        stats["avg_entities_per_doc"] = stats["total_entities"] / stats["total_documents"]
    
    print(f"   Статистика: {stats['total_documents']} документов, {stats['total_entities']} сущностей")
    print(f"   Типы сущностей: {list(entity_counter.keys())}")
    
    return stats


def setup_auto_retraining_system(reference_stats_dir: str, config_path: str = "config/config.yaml") -> DriftDetector:
    """Настройка системы автоматического переобучения"""
    print(f"[SETUP] Computing reference statistics from {reference_stats_dir}...")
    reference_stats = compute_data_statistics(reference_stats_dir)
    
    # Создаём папку data/reference если её нет
    ref_dir = Path("data/reference")
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    ref_stats_path = ref_dir / "reference_stats.json"
    with open(ref_stats_path, 'w', encoding='utf-8') as f:
        json.dump(reference_stats, f, indent=2, ensure_ascii=False)
    
    print(f"[SETUP] Reference statistics saved to {ref_stats_path}")
    
    detector = DriftDetector(
        reference_stats_path=str(ref_stats_path),
        threshold_js_divergence=0.15,
        threshold_new_tokens_ratio=0.3,
        min_samples_for_detection=50
    )
    
    return detector

def log_retrain_event(event_type: str, details: dict):
    """Запись события переобучения в лог"""
    log_path = Path("logs/retrain_records.json")
    
    if log_path.exists():
        with open(log_path, 'r') as f:
            records = json.load(f)
    else:
        records = []
    
    records.append({
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details
    })
    
    with open(log_path, 'w') as f:
        json.dump(records, f, indent=2)


def scheduled_drift_check(new_data_dir: str, auto_retrainer: DriftDetector, force_retrain: bool = False) -> Dict[str, Any]:
    """Периодическая проверка дрейфа"""
    print(f"[SCHEDULED] Checking new data in {new_data_dir}")
    
    new_stats = compute_data_statistics(new_data_dir)
    
    if new_stats['total_documents'] == 0:
        print("[SCHEDULED] No new data to analyze")
        return {"status": "no_data", "message": "No new documents"}
    
    print(f"[SCHEDULED] Found {new_stats['total_documents']} new documents")
    
    has_drift, drift_metrics = auto_retrainer.detect_drift(new_stats)
    
    # Логируем проверку дрейфа
    log_retrain_event("drift_check", {
        "has_drift": has_drift,
        "js_divergence": drift_metrics.get("js_divergence"),
        "new_tokens_ratio": drift_metrics.get("new_tokens_ratio")
    })
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "new_data_stats": {
            "documents": new_stats['total_documents'],
            "entities": new_stats['total_entities'],
        },
        "result": {
            "has_drift": has_drift,
            "drift_metrics": drift_metrics,
            "retrained": False
        }
    }
    
    report_dir = Path("reports/monitoring")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"drift_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"[SCHEDULED] Report saved to {report_path}")
    
    return result
