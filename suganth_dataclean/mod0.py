import os
import json
import pandas as pd
from collections import defaultdict

# Team name mapping
TEAM_NAME_MAP = {
    'Ben': 'Bengaluru Bulls',
    'Kol': 'Bengal Warriors',
    'Dab': 'Dabang Delhi K.C.',
    'GFG': 'Gujarat Fortunegiants',
    'Hyd': 'Telugu Titans',
    'Del': 'Dabang Delhi K.C.',
    'HS': 'Haryana Steelers',
    'Jai': 'Jaipur Pink Panthers',
    'Jaipur': 'Jaipur Pink Panthers',
    'Pat': 'Patna Pirates',
    'Pun': 'Puneri Paltan',
    'TT': 'Tamil Thalaivas',
    'Mum': 'U Mumba',
    'UPY': 'U.P. Yoddha',
}

def process_single_folder(base_dir, stat_name, data_type):

    data_list = []
    for season_num in range(1, 8):
        season_file = os.path.join(base_dir, f'Season_{season_num}.json')
        if not os.path.exists(season_file):
            print(f"Warning: JSON file not found for {season_file}. Skipping.")
            continue
        try:
            with open(season_file, 'r', encoding='utf-8') as f:
                season_data = json.load(f)
            if "data" in season_data and isinstance(season_data["data"], list):
                for item in season_data["data"]:
                    if isinstance(item, dict):
                        stats = defaultdict(lambda: None)
                        stats['Season'] = season_num
                        stats[stat_name] = item.get('value')
                        
                        if data_type == 'player':
                            stats['player_name'] = item.get('player_name') or item.get('player')
                            stats['team_name'] = item.get('team_name') or item.get('team')
                            stats['player_id'] = item.get('player_id')
                            stats['match_played'] = item.get('match_played')
                            stats['position_id'] = item.get('position_id')
                            stats['position_name'] = item.get('position_name')
                        elif data_type == 'team':
                            stats['team_name'] = item.get('team_name') or item.get('team')
                            stats['team_id'] = item.get('team_id')
                            stats['match_played'] = item.get('match_played')

                        data_list.append(dict(stats))
                    else:
                        print(f"Warning: Found non-dictionary item in JSON list for {season_file}. Skipping item: {item}")
            else:
                print(f"Error: JSON file for {season_file} does not contain a 'data' key with a list. Skipping.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {season_file}. Skipping.")
        except KeyError as e:
            print(f"Missing key in JSON file: {season_file}. Error: {e}. Skipping.")
    return data_list

def process_and_standardize(data_list, base_dirs_map, output_filename, data_type):

    if not data_list:
        print(f"No {data_type} data was found. Please check your folder structure and JSON file contents.")
        return None

    df = pd.DataFrame(data_list)
    print(f"\nStarting data cleaning for {data_type} data...")
    df.columns = [col.lower() for col in df.columns]

    sort_key = 'player_name' if data_type == 'player' else 'team_name'
    df.sort_values(by=[sort_key, 'season'], ascending=True, inplace=True)
    df = df.groupby(sort_key).last().reset_index()


    df['team_name'] = df['team_name'].str.strip().replace(TEAM_NAME_MAP)

    numeric_cols = ['match_played'] + list(base_dirs_map.values())
    if data_type == 'player':
        numeric_cols.append('position_id')
    else:
        numeric_cols.append('team_id')

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print("Data cleaning completed.")

    print("Starting normalization and standardization...")
    standardization_cols = list(base_dirs_map.values())
    for col in standardization_cols:
        if df[col].std() > 0:
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[f'z_score_{col}'] = (df[col] - mean_val) / std_val
        else:
            df[f'z_score_{col}'] = 0
    print("Normalization and standardization completed.")

    df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully processed {data_type} data and saved to '{output_filename}'.")
    return df

if __name__ == "__main__":

    player_dirs_map = {
        'Player_do_or_die': 'do_or_die_points',
        'Player_high_5s': 'high_5s',
        'Player_avg_raid_points': 'avg_raid_points',
        'Player_raidpoints': 'raid_points',
        'Player_successful_raids' : 'successful_raids',
        'Player_successful_tackles':'successful_tackles',
        'Player_super_10s':'super_10s',
        'Player_super_raids':'super_raids',
        'Player_super_takels':'super_tackles',
        'Player_tackle_points':'tackle_points',
        'Player_Total_points':'total_points'
    }
    
    team_dirs_map = {
        'Team_Allouts_conceded': 'allouts_conceded',
        'Team_Allouts_inflicted': 'allouts_inflicted',
        'Team_avg_points_scored': 'avg_points_scored',
        'Team_avg_raid_points': 'avg_raid_points',
        'Team_avg_tackle_points': 'avg_tackle_points',
        'Team_conceded_points': 'conceded_points',
        'Team_do_die_points': 'do_or_die_points',
        'Team_points_scored': 'points_scored',
        'Team_raid_points': 'raid_points',
        'Team_successful_raids': 'successful_raids',
        'Team_successful_tackles': 'successful_tackles',
        'Team_super_raids': 'super_raids',
        'Team_super_tackles': 'super_tackles',
        'Team_tackle_points': 'tackle_points'
    }

    all_player_data = []
    for folder, stat_name in player_dirs_map.items():
        base_dir = f'./{folder}'
        print(f"\nProcessing player folder: {base_dir} for statistic: {stat_name}")
        all_player_data.extend(process_single_folder(base_dir, stat_name, 'player'))
    processed_player_df = process_and_standardize(all_player_data, player_dirs_map, 'processed_kabaddi_stats.csv', 'player')
    if processed_player_df is not None:
        print("\nFinal Processed Player DataFrame:")
        print(processed_player_df)

    all_team_data = []
    for folder, stat_name in team_dirs_map.items():
        base_dir = f'./{folder}'
        print(f"\nProcessing team folder: {base_dir} for statistic: {stat_name}")
        all_team_data.extend(process_single_folder(base_dir, stat_name, 'team'))
    processed_team_df = process_and_standardize(all_team_data, team_dirs_map, 'processed_kabaddi_teams_stats.csv', 'team')
    if processed_team_df is not None:
        print("\nFinal Processed Team DataFrame:")
        print(processed_team_df)
