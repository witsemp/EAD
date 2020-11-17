import sqlite3
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_pokemon_info(pokemon_name: str, pokemon_stat_dict: dict):
    base_url = "https://pokeapi.co/api/v2/pokemon/"
    pokemon_url = base_url + pokemon_name.lower()
    req = requests.get(pokemon_url)
    if req.status_code != 404:
        pokemon = json.loads(req.text)
        pokemon_stat_dict[pokemon_name] = {}
        pokemon_stat_dict[pokemon_name]['hp'] = pokemon['stats'][0]['base_stat']
        pokemon_stat_dict[pokemon_name]['attack'] = pokemon['stats'][1]['base_stat']
        pokemon_stat_dict[pokemon_name]['defense'] = pokemon['stats'][2]['base_stat']
        pokemon_stat_dict[pokemon_name]['special-attack'] = pokemon['stats'][3]['base_stat']
        pokemon_stat_dict[pokemon_name]['special-defense'] = pokemon['stats'][4]['base_stat']
        pokemon_stat_dict[pokemon_name]['speed'] = pokemon['stats'][5]['base_stat']
        if len(pokemon['types']) == 2:
            pokemon_stat_dict[pokemon_name]['type_1'] = pokemon['types'][0]['type']['name']
            pokemon_stat_dict[pokemon_name]['type_2'] = pokemon['types'][1]['type']['name']
        else:
            pokemon_stat_dict[pokemon_name]['type_1'] = pokemon['types'][0]['type']['name']
            pokemon_stat_dict[pokemon_name]['type_2'] = None


def attack_against(attacker: str, attacked: str, database: pd.DataFrame):
    attacked_type1 = database.loc[attacked, ['type_1']].iloc[0]
    attacked_type2 = database.loc[attacked, ['type_2']].iloc[0]
    att_ag_1 = 'against_' + attacked_type1
    if attacked_type2 is not None:
        att_ag_2 = 'against_' + attacked_type2
        attack_value = database.loc[attacker, ['attack']].iloc[0] * database.loc[attacker, [att_ag_1]].iloc[0] * \
                       database.loc[
                           attacker, [att_ag_2]].iloc[0]
        print("Attack value: ", attack_value)
        return attack_value

    else:
        attack_value = database.loc[attacker, ['attack']].iloc[0] * database.loc[attacker, [att_ag_1]].iloc[0]
        print("Attack value: ", attack_value)
        return attack_value


def main():
    pokemon_stat_dict = {}
    conn = sqlite3.connect("pokemon_against.sqlite")
    pokemon_encounter_df = pd.read_hdf("pokedex_history.hdf5")
    for name in pokemon_encounter_df['name']:
        get_pokemon_info(name, pokemon_stat_dict)
    pokemon_stat_df = pd.DataFrame.from_dict(pokemon_stat_dict, orient='index')
    pokemon_stat_df = pokemon_stat_df.rename_axis('name')
    against_df = pd.read_sql_query('SELECT * FROM against_stats', conn)
    against_df = against_df.set_index('name')
    merged_df = pd.merge(pokemon_stat_df, against_df, on='name', how='inner')
    print(merged_df)
    attack_against('Slurpuff', 'Roselia', merged_df)


if __name__ == '__main__':
    main()
