"""
Модуль для визуализации данных и метрик.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


class NERVisualizer:
    """
    Визуализатор для NER данных.
    """
    
    def __init__(self, output_dir: str = "reports/visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ANSI_COLORS = {
            'COUNTRY': '\033[91m',
            'CITY': '\033[96m',
            'LOCATION': '\033[94m',
            'STATE_OR_PROV': '\033[92m',
            'DISTRICT': '\033[93m',
        }
        self.RESET = '\033[0m'
    
    def get_color_for_type(self, entity_type: str) -> str:
        return self.ANSI_COLORS.get(entity_type, self.RESET)
    
    def print_legend(self) -> None:
        """
        Выводит легенду типов сущностей с соответствующими цветами.
        """
        print("\nЛегенда типов сущностей:")
        for ent_type, color in self.ANSI_COLORS.items():
            print(f"  {color}{ent_type}{self.RESET}")
        print()
    
    def highlight_entities_terminal(self, text: str, entities: List[Dict[str, Any]], show_legend: bool = True) -> str:
        """
        Возвращает текст с ANSI-цветами для терминала.
        
        Args:
            text: Исходный текст
            entities: Список сущностей с полями 'start', 'end', 'type'
            show_legend: Показать легенду перед выводом
        
        Returns:
            Текст с ANSI-кодами цветов
        """
        if show_legend:
            self.print_legend()
        
        if not entities:
            return text
        
        sorted_entities = sorted(entities, key=lambda e: e.get('start', 0), reverse=True)
        
        result = text
        for ent in sorted_entities:
            start = ent.get('start', 0)
            end = ent.get('end', len(ent.get('text', '')))
            ent_type = ent.get('type', 'LOCATION')
            
            if start < len(result):
                color = self.ANSI_COLORS.get(ent_type, self.RESET)
                highlighted = f"{color}{result[start:end]}{self.RESET}"
                result = result[:start] + highlighted + result[end:]
        
        return result
    
    def plot_entity_distribution(self, entity_counts: Dict[str, int], title: str = "Распределение типов сущностей", save: bool = True, open_result: bool = False) -> Path:
        types = list(entity_counts.keys())
        counts = list(entity_counts.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(types, counts, color='skyblue', edgecolor='black')
        plt.xlabel('Тип сущности')
        plt.ylabel('Количество')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.output_dir / "entity_distribution.png"
        if save:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  График сохранён: {save_path}")
            return save_path
        
        if open_result:
            plt.show()
        
        return save_path
    
    def plot_consistency_over_batches(self, batch_consistencies: List[Dict[str, Any]], title: str = "Динамика консистентности по батчам", save: bool = True, open_result: bool = False) -> Path:
        batches = [c.get('batch_num', i+1) for i, c in enumerate(batch_consistencies)]
        ratios = [c.get('consistency_ratio', 0) for c in batch_consistencies]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batches, ratios, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('Номер батча')
        plt.ylabel('Доля консистентных сущностей')
        plt.title(title)
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "consistency_over_batches.png"
        if save:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  График сохранён: {save_path}")
            return save_path
        
        if open_result:
            plt.show()
        
        return save_path
    
    def plot_filter_impact(self, before_counts: Dict[str, int], after_counts: Dict[str, int], title: str = "Распределение сущностей до и после фильтрации", save: bool = True, open_result: bool = False) -> Path:
        all_types = sorted(set(before_counts.keys()) | set(after_counts.keys()))
        before_vals = [before_counts.get(t, 0) for t in all_types]
        after_vals = [after_counts.get(t, 0) for t in all_types]
        
        x = range(len(all_types))
        width = 0.35
        
        plt.figure(figsize=(14, 7))
        plt.bar([i - width/2 for i in x], before_vals, width, label='До фильтрации', color='skyblue', edgecolor='black')
        plt.bar([i + width/2 for i in x], after_vals, width, label='После фильтрации', color='lightcoral', edgecolor='black')
        
        plt.xlabel('Тип сущности')
        plt.ylabel('Количество')
        plt.title(title)
        plt.xticks(x, all_types, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        save_path = self.output_dir / "filter_impact.png"
        if save:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  График сохранён: {save_path}")
            return save_path
        
        if open_result:
            plt.show()
        
        return save_path
    
    def plot_length_distribution(self, lengths: List[int], title: str = "Распределение длины документов", save: bool = True, open_result: bool = False) -> Path:
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Длина документа (символы)')
        plt.ylabel('Частота')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "length_distribution.png"
        if save:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  График сохранён: {save_path}")
            return save_path
        
        if open_result:
            plt.show()
        
        return save_path