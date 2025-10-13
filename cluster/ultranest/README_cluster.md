# Running UltraNest on PDC/NAISS Cluster

## Changes Made for Cluster Compatibility

1. **Dynamic Path Resolution**: Replaced hard-coded paths with relative paths using `pathlib.Path`
2. **Non-interactive Plotting**: Added matplotlib backend configuration for headless environments
3. **Output Management**: Results are saved to files instead of displayed interactively
4. **Command Line Interface**: Added argument parsing for flexible execution
5. **Resource Management**: Added support for limiting function calls for time-constrained jobs

## Setup Instructions

1. **Load Required Modules**:
   ```bash
   module load Python/3.9.6-GCCcore-11.2.0
   module load matplotlib/3.5.1-foss-2021b
   module load scipy-bundle/2021.10-foss-2021b
   ```

2. **Install Dependencies** (if needed):
   ```bash
   pip install --user -r requirements.txt
   ```

3. **Submit Job**:
   ```bash
   sbatch run_ultranest_job.sh
   ```

## Usage Options

### Basic Run:
```bash
python3 hese12_parameter_fit.py
```

### With Custom Output Directory:
```bash
python3 hese12_parameter_fit.py --output-dir /path/to/results
```

### With Time Limit (for shorter jobs):
```bash
python3 hese12_parameter_fit.py --max-ncalls 50000
```

## Output Files

The script will create the following files in the output directory:
- `results.json`: Complete UltraNest results
- `summary.txt`: Human-readable summary
- `corner_plot.png`: Parameter correlation plot
- `ultranest.log`: Detailed log file

## SLURM Job Script

The provided `run_ultranest_job.sh` script includes:
- 24-hour time limit
- 8 CPU cores
- 32GB memory
- Automatic output directory creation with timestamp
- Proper module loading for PDC/NAISS

## Troubleshooting

1. **Module Issues**: Check available modules with `module avail`
2. **Path Issues**: Ensure all data files are in the correct relative locations
3. **Memory Issues**: Reduce `--max-ncalls` or increase memory allocation
4. **Time Issues**: Use `--max-ncalls` to limit computation time

## Notes

- The script now uses relative paths, so it should work regardless of where it's run from
- All plotting is non-interactive and saves to files
- Results are automatically saved for later analysis
- The script can be interrupted and resumed (UltraNest supports this)






