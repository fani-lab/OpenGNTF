import pandas as pd
import numpy as np
import scipy.sparse



a = pd.read_pickle('../data/teamsvecs.pkl')

ids = np.asarray(a['id'].todense())
idss = []
for i in ids:
    idss.append(i.item()-1)

skills = np.asarray(a['skill'].todense())
members = np.asarray(a['member'].todense())


# Convert one-hot vectors to lists of indices
required_skills = [np.where(row == 1)[0].tolist() for row in skills]
team_members = [np.where(row == 1)[0].tolist() for row in members]

# Create the Teams dataframe
teams_df = pd.DataFrame({'team_id': idss, 'required_skills': required_skills, 'members': team_members})

# Save the Teams dataframe to CSV
teams_df.to_csv('Teams.csv', index=False)

# Build user_teams_dict based on the Teams dataframe
user_teams_dict = {}
for _, row in teams_df.iterrows():
    team_id = row['team_id']
    members = row['members']

    for member in members:
        user_id = member  # Assuming user ids are like 'u1', 'u2', ...

        if user_id not in user_teams_dict:
            user_teams_dict[user_id] = [team_id]
        else:
            user_teams_dict[user_id].append(team_id)

# Print the resulting user_teams_dict
print(user_teams_dict)

# Aggregate skills for each user
user_skills_dict = {}
for user_id, teams in user_teams_dict.items():
    user_skills = []
    for team_id in teams:
        # team_index = np.where(idss == team_id)[0][0]
        team_skills = np.where(skills[team_id] == 1)[0].tolist()
        user_skills.extend(team_skills)
    user_skills_dict[user_id] = list(set(user_skills))

# Create the Users dataframe
users_df = pd.DataFrame({'user_id': list(user_skills_dict.keys()), 'skills': list(user_skills_dict.values())})

# Save the Users dataframe to CSV
users_df.to_csv('Users.csv', index=False)