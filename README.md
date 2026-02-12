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
│   ├── [1-22]/           # Period Number (1-2: Practice, 3-22: Experiment)
│   │   ├── 1_market.json      # Market-level data/snapshots
│   │   └── 1_participant.json # Participant-level data/snapshots
├── flow[1-10]/
│   ├── [1-22]/
│   │   ├── 1_market.json
│   │   └── 1_participant.json
```

## JSON File Field Descriptions

### Market Data (`1_market.json`)
Contains snapshots of the market state for every second of the period.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | Integer | Time in seconds from the start of the period (0-120). |
| `id_in_subsession` | Integer | Unique identifier for the subsession (period). |
| `before_transaction` | Boolean | `true` indicates state before transaction processing in that second; `false` indicates state after. |
| `clearing_price` | Float/Null | The market clearing price. `null` if no clearing occurred. |
| `clearing_rate` | Float/Null | The volume (CDA) or rate (Flow) of clearing. |

### Participant Data (`1_participant.json`)
Contains snapshots of each participant's state for every second.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | Integer | Time in seconds. |
| `id_in_subsession` | Integer | Unique identifier for the subsession. |
| `id_in_group` | Integer | Participant ID within the group (1-8). |
| `participant_id` | Integer | Unique database ID for the participant. |
| `before_transaction` | Boolean | `true` indicates state before transaction processing in that second. |
| `active_orders` | Array | List of active orders placed by the participant. |
| `active_contracts` | Array | List of active contracts held. |
| `executed_contracts` | Array | List of contracts executed by the participant. |
| `cash` | Float | Current cash balance. |
| `inventory` | Float | Current share inventory (positive or negative). |
| `rate` | Float | (Flow Only) Current trading rate set by the participant. |

## Experimental Configuration

(Derived from `config.py`)

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

Analysis scripts (e.g., `flow.py`, `cda.py`) read the JSON files and generate aggregated CSV files:
- `data_interval.csv`
- `data_period.csv`
- `data_second.csv`
- `data_profits.csv`
- `data_liquidity.csv`
