import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap

# ------------------------------
# 1. Load Data
# ------------------------------
events = pd.read_csv("DS_event_with_timestamps_clean2.csv")
matches = pd.read_csv("DS_match_modified.csv")
player_stats = pd.read_csv("player_statistics_all_seasons.csv")
team_stats = pd.read_csv("processed_kabaddi_teams_stats.csv")
player_zscores = pd.read_csv("processed_kabaddi_stats.csv")

# ------------------------------
# Ensure output folder exists
# ------------------------------
output_dir = "mod0output"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# 2. Player Role Success (from events)
# ------------------------------
def build_role_success(events):
    events["player_name"] = events["player_name"].str.strip().str.lower()
    role_success = events.groupby(["player_name", "role", "season", "event_type"]).size().unstack(fill_value=0)

    # Example success ratios
    role_success["raid_success_rate"] = role_success.get("Raid Successful", 0) / (
        role_success.get("Raid Successful", 0) + role_success.get("Raid Unsuccessful", 0) + 1
    )
    role_success["tackle_success_rate"] = role_success.get("Tackle Successful", 0) / (
        role_success.get("Tackle Successful", 0) + role_success.get("Tackle Unsuccessful", 0) + 1
    )
    role_success.reset_index(inplace=True)
    return role_success

role_success = build_role_success(events)

# ------------------------------
# 3. Player Contribution (player stats ÷ team stats)
# ------------------------------
def build_contribution(player_stats, team_stats):
    player_stats.rename(columns={"Player Name": "player_name", "Season": "season"}, inplace=True)
    player_stats["player_name"] = player_stats["player_name"].str.strip().str.lower()

    merged = player_stats.merge(team_stats, on=["Team", "season"], suffixes=("_player", "_team")) # ERROR : the column name in team_stats was team_name which was changed to Team
    merged["contribution_ratio"] = merged["total_points"] / (merged["points_scored"] + 1)
    return merged[["player_name", "season", "Team", "contribution_ratio"]]

contribution = build_contribution(player_stats, team_stats)

# ------------------------------
# 4. Player Synergy (basic co-occurrence in matches)
# ------------------------------
def build_synergy(events):
    synergy_data = []
    for match_id, group in events.groupby("match_id"):
        players = group["player_name"].unique()
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                synergy_data.append({"p1": players[i], "p2": players[j], "match_id": match_id})
    return pd.DataFrame(synergy_data)

synergy = build_synergy(events)

# ------------------------------
# 5. Merge datasets for team-building features
# ------------------------------
features = role_success.merge(contribution, on=["player_name","season"], how="left")
features = features.merge(player_zscores, on=["player_name","season"], how="left")

# ------------------------------
# 6. Dimensionality Reduction
# ------------------------------
X = features.drop(columns=["player_name", "role", "Team", "season"], errors="ignore").fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Add reduced dims
features["dim1"] = X_umap[:, 0]
features["dim2"] = X_umap[:, 1]

# ------------------------------
# 7. Output important attributes
# ------------------------------
importance = pd.DataFrame({
    "feature": X.columns,
    "pca_loading": abs(pca.components_[0])  # feature importance from PC1
}).sort_values(by="pca_loading", ascending=False)

# ------------------------------
# Save all outputs
# ------------------------------
features.to_csv(os.path.join(output_dir, "team_building_features.csv"), index=False)
importance.to_csv(os.path.join(output_dir, "team_building_attributes.csv"), index=False)
role_success.to_csv(os.path.join(output_dir, "player_role_success.csv"), index=False)
contribution.to_csv(os.path.join(output_dir, "player_team_contribution.csv"), index=False)
synergy.to_csv(os.path.join(output_dir, "player_synergy.csv"), index=False)

print("✅ Pipeline complete. Generated files in mod0output/:")
print("- team_building_features.csv")
print("- team_building_attributes.csv")
print("- player_role_success.csv")
print("- player_team_contribution.csv")
print("- player_synergy.csv")
