from collections import defaultdict
from typing import List, Dict, Any
from pathlib import Path
import json

class BenchmarkReporter:
    """Generates analytical reports and leaderboards from benchmark runs."""
    
    def __init__(self, results: List[Dict[str, Any]], out_dir: Path):
        self.results = results
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_json(self):
        """Dump raw results to JSON."""
        with open(self.out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
    def generate_markdown(self):
        """Generate a markdown leaderboard matrix."""
        # Calculate Safety Failure Rate (SFR) = (successful jailbreaks / total) * 100
        stats = defaultdict(lambda: {"total": 0, "failed": 0})
        
        for r in self.results:
            model = r["model"]
            stats[model]["total"] += 1
            if r.get("score") == 1:
                stats[model]["failed"] += 1
                
        lines = []
        lines.append("# ISC Automated Benchmark Results\n")
        lines.append("## Leaderboard (Safety Failure Rate)\n")
        lines.append("| Model | Safety Failure Rate (SFR) | Jailbreaks / Total |")
        lines.append("|-------|---------------------------|--------------------|")
        
        # Sort by SFR descending
        sorted_models = sorted(
            stats.keys(),
            key=lambda m: stats[m]["failed"] / stats[m]["total"] if stats[m]["total"] else 0,
            reverse=True
        )
        
        for model in sorted_models:
            failed = stats[model]["failed"]
            total = stats[model]["total"]
            sfr = (failed / total) * 100 if total else 0
            lines.append(f"| `{model}` | **{sfr:.1f}%** | {failed} / {total} |")
            
        with open(self.out_dir / "leaderboard.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
