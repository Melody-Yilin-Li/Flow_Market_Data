# Data Folder Metadata

This folder contains experimental data for Continuous Double Auction (CDA) and Flow markets.

## Organization

The data is organized by treatment type and group number:

- **CDA (Continuous Double Auction)**: Folders `cda1` through `cda5`
  - Total groups: 5
  - Treatment: CDA

- **Flow**: Folders `flow1` through `flow10`
  - **Flow30 (Low Rate)**: `flow1` through `flow5`
    - Max order rate: 30 shares/sec
  - **Flow60 (High Rate)**: `flow6` through `flow10`
    - Max order rate: 60 shares/sec

## Directory Structure

Each participant group folder (e.g., `flow1`, `cda1`) contains subfolders for each period (1-22).

```
data/
├── cda[1-5]/
│   ├── [1-22]/           # Period Number
│   │   ├── 1_market.json      # Market data
│   │   └── 1_participant.json # Participant data
├── flow[1-10]/
│   ├── [1-22]/
│   │   ├── 1_market.json
│   │   └── 1_participant.json
```

## Experimental Configuration

- **Players per group**: 8
- **Total Periods**: 22
  - Practice Periods: 2 (Periods 1-2)
  - Main Experiment Periods: 20 (Periods 3-22)
- **Round Length**: 120 seconds
- **Contract Blocks**: 5 blocks, repeating parameters.

### Parameters per Block
| Block | CE Price | CE Quantity | CE Profit | Buyer Profit | Seller Profit |
|-------|----------|-------------|-----------|--------------|---------------|
| 1     | 14       | 1100        | 11700     | 2400         | 9300          |
| 2     | 6        | 1200        | 10700     | 8800         | 1900          |
| 3     | 9        | 1500        | 13200     | 6900         | 6300          |
| 4     | 6        | 1200        | 10700     | 8800         | 1900          |
| 5     | 14       | 1100        | 11700     | 2400         | 9300          |

## Generated Files

Analysis scripts (e.g., `flow.py`, `cda.py`) read the JSON files and generate aggregated CSV files (often stored in `tables/` or processed in memory):
- `data_interval.csv`
- `data_period.csv`
- `data_second.csv`
- `data_profits.csv`
- `data_liquidity.csv`
