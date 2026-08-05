import argparse
import asyncio
from pathlib import Path
from core.engine import BenchmarkEngine
from core.reporter import BenchmarkReporter

async def main():
    parser = argparse.ArgumentParser(description="ISC Automated Benchmarking Suite")
    parser.add_argument("--config", type=str, default="benchmark_config.yaml", help="Path to YAML config file")
    parser.add_argument("--outdir", type=str, default="reports", help="Output directory for reports")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {args.config}")
        return
        
    print(f"Initializing BenchmarkEngine with {args.config}...")
    engine = BenchmarkEngine(args.config)
    
    print("Running scenarios...")
    results = await engine.run_all()
    
    out_dir = config_path.parent / args.outdir
    print(f"Generating reports in {out_dir}...")
    reporter = BenchmarkReporter(results, out_dir)
    reporter.generate_json()
    reporter.generate_markdown()
    print("Done! Check leaderboard.md for results.")

if __name__ == "__main__":
    asyncio.run(main())
