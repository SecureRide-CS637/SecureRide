#!/usr/bin/env python3
"""
Extracts realistic sample telemetry files from held-out trips for sample inference testing.
"""
import os
import pandas as pd

def main():
    trips_path = "data/trips.csv"
    if not os.path.exists(trips_path):
        print(f"Error: {trips_path} not found.")
        return

    df = pd.read_csv(trips_path)
    # Use test trips to ensure held-out validation
    test_df = df[df.split == "test"]

    # 1. Sample Speeding Illegal: sustained speed > 21 km/h for 75 slots (15s)
    speeding_trip = None
    for trip_id, g in test_df.groupby("trip_id"):
        fast_idx = g.index[g.speed > 21.0].tolist()
        if len(fast_idx) >= 75:
            # Find contiguous block
            diffs = [fast_idx[i+1] - fast_idx[i] for i in range(len(fast_idx)-1)]
            if diffs and max(diffs[:74]) == 1:
                speeding_trip = g.loc[fast_idx[:75]].copy()
                break
    if speeding_trip is None:
        speeding_trip = test_df[test_df.speed > 20.0].head(75).copy()

    # 2. Sample Speeding Legal: sustained speed between 10 and 16 km/h for 75 slots
    legal_speed_trip = None
    for trip_id, g in test_df.groupby("trip_id"):
        leg_idx = g.index[(g.speed >= 10.0) & (g.speed <= 16.0)].tolist()
        if len(leg_idx) >= 75:
            legal_speed_trip = g.loc[leg_idx[:75]].copy()
            break
    if legal_speed_trip is None:
        legal_speed_trip = test_df[(test_df.speed >= 10.0) & (test_df.speed <= 16.0)].head(75).copy()

    # 3. Sample Sidewalk Illegal: surface > 0 for 75 slots
    sidewalk_trip = None
    for trip_id, g in test_df.groupby("trip_id"):
        sw_idx = g.index[g.surface > 0].tolist()
        if len(sw_idx) >= 75:
            sidewalk_trip = g.loc[sw_idx[:75]].copy()
            break
    if sidewalk_trip is None:
        sidewalk_trip = test_df[test_df.surface > 0].head(75).copy()

    # 4. Sample Sidewalk Legal: surface == 0 (bicycle road) for 75 slots
    road_trip = None
    for trip_id, g in test_df.groupby("trip_id"):
        rd_idx = g.index[g.surface == 0].tolist()
        if len(rd_idx) >= 75:
            road_trip = g.loc[rd_idx[:75]].copy()
            break
    if road_trip is None:
        road_trip = test_df[test_df.surface == 0].head(75).copy()

    # 5. Mixed Trip: 150 slots (30s) containing transition
    mixed_trip = test_df[test_df.trip_id == test_df.trip_id.iloc[0]].head(150).copy()

    cols = ["voltage", "current", "capacity"]
    # Save standard telemetry files containing battery channels
    speeding_trip[cols].reset_index(drop=True).to_csv("sample_speeding_illegal.csv", index=False)
    legal_speed_trip[cols].reset_index(drop=True).to_csv("sample_speeding_legal.csv", index=False)
    sidewalk_trip[cols].reset_index(drop=True).to_csv("sample_sidewalk_illegal.csv", index=False)
    road_trip[cols].reset_index(drop=True).to_csv("sample_sidewalk_legal.csv", index=False)
    mixed_trip[cols].reset_index(drop=True).to_csv("sample_mixed_trip.csv", index=False)

    print("Created sample CSV files:")
    print("  - sample_speeding_illegal.csv")
    print("  - sample_speeding_legal.csv")
    print("  - sample_sidewalk_illegal.csv")
    print("  - sample_sidewalk_legal.csv")
    print("  - sample_mixed_trip.csv")

if __name__ == "__main__":
    main()
