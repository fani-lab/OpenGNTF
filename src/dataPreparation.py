import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

def main(experts_df, teams_df, path, full_subgraph="", graph_type="STE"):
    # Update skillset and required_skillset with string representation
    experts_df['skillset'] = [str(row) for row in experts_df['skills']]
    teams_df['required_skillset'] = [str(row) for row in teams_df["required_skills"]]
    teams_df['location'] = [row[0] for row in teams_df["location"]]

    # Create skill mapping
    all_skills = set(skill for skills_list in experts_df['skills'] for skill in skills_list)
    skills_df = pd.DataFrame(list(enumerate(all_skills, start=0)), columns=["skill_id", "skill_name"])
    skill_mapping = {skill_name: skill_id for skill_id, skill_name in zip(skills_df['skill_id'], skills_df['skill_name'])}
    experts_df['skills'] = experts_df['skills'].apply(lambda skills_list: [skill_mapping[skill] for skill in skills_list])
    teams_df['required_skillset'] = teams_df['required_skills'].apply(lambda skills_list: [skill_mapping[skill] for skill in skills_list])

    # Create edge index for expert-skill
    edges_expert_skill = []
    for _, row in tqdm(experts_df.iterrows(), desc="Processing experts"):
        expert_index = row['user_id']
        for skillID in row['skills']:
            skill_index = skillID
            edges_expert_skill.append((expert_index, skill_index))
    edge_index_expert_skill = torch.tensor(list(zip(*edges_expert_skill)), dtype=torch.long)

    if graph_type == "SE":
        data = HeteroData()
        data['expert'].node_id = torch.tensor(experts_df["user_id"].values, dtype=torch.long)
        data['skill'].node_id = torch.tensor(skills_df["skill_id"].values, dtype=torch.long)
        data['expert', 'has', 'skill'].edge_index = edge_index_expert_skill
        data['expert', 'has', 'skill'].edge_attr = None

        if not full_subgraph == "":
            expert_pairs = []
            skill_pairs = []
            expert_skill_pairs = []
            for _, row in tqdm(teams_df.iterrows(), desc="Processing full subgraphs"):
                experts = row['members']
                skills = row['required_skillset']
                for i in range(len(experts)):
                    for j in range(i + 1, len(experts)):
                        expert_pairs.append((experts[i], experts[j]))
                        expert_pairs.append((experts[j], experts[i]))
                for i in range(len(skills)):
                    for j in range(i + 1, len(skills)):
                        skill_pairs.append((skills[i], skills[j]))
                        skill_pairs.append((skills[j], skills[i]))
                for expert in experts:
                    for skill in skills:
                        expert_skill_pairs.append((expert, skill))
                        expert_skill_pairs.append((skill, expert))
            expert_pairs = list(set(expert_pairs))
            skill_pairs = list(set(skill_pairs))
            expert_skill_pairs = list(set(expert_skill_pairs))

            if expert_pairs:
                data['expert', 'connected_to', 'expert'].edge_index = torch.tensor(list(zip(*expert_pairs)), dtype=torch.long)
            if skill_pairs:
                data['skill', 'connected_to', 'skill'].edge_index = torch.tensor(list(zip(*skill_pairs)), dtype=torch.long)
            if expert_skill_pairs:
                data['expert', 'connected_to', 'skill'].edge_index = torch.tensor(list(zip(*expert_skill_pairs)), dtype=torch.long)
    else:
        edges_team_skill = []
        for _, row in tqdm(teams_df.iterrows()):
            team_index = row['team_id']
            for skillID in row['required_skillset']:
                skill_index = skillID
                edges_team_skill.append((team_index, skill_index))
        edge_index_team_skill = torch.tensor(list(zip(*edges_team_skill)), dtype=torch.long)

        edges_team_experts = []
        for _, row in tqdm(teams_df.iterrows()):
            team_index = row['team_id']
            for expertID in row['members']:
                expert_index = expertID
                edges_team_experts.append((team_index, expert_index))
        edge_index_team_experts = torch.tensor(list(zip(*edges_team_experts)), dtype=torch.long)

        if graph_type == "STEL":
            edges_team_location = []
            for _, row in tqdm(teams_df.iterrows()):
                team_index = row['team_id']
                location_index = row['location']
                edges_team_location.append((team_index, location_index))
            edge_index_team_location = torch.tensor(list(zip(*edges_team_location)), dtype=torch.long)

        data = HeteroData()
        data['expert'].node_id = torch.tensor(experts_df["user_id"].values, dtype=torch.long)
        data['skill'].node_id = torch.tensor(skills_df["skill_id"].values, dtype=torch.long)
        data['team'].node_id = torch.tensor(teams_df["team_id"].values, dtype=torch.long)
        data['team', 'requires', 'skill'].edge_index = edge_index_team_skill
        data['team', 'includes', 'expert'].edge_index = edge_index_team_experts
        data['expert', 'has', 'skill'].edge_index = edge_index_expert_skill
        data['team', 'requires', 'skill'].edge_attr = None
        data['team', 'includes', 'expert'].edge_attr = None
        data['expert', 'has', 'skill'].edge_attr = None

        if graph_type == "STEL":
            data['location'].node_id = torch.tensor(teams_df["location"].values, dtype=torch.long)
            data['team', 'located_at', 'location'].edge_index = edge_index_team_location
            data['team', 'located_at', 'location'].edge_attr = None

        if not full_subgraph == "":
            expert_pairs = []
            skill_pairs = []
            expert_skill_pairs = []
            for _, row in tqdm(teams_df.iterrows(), desc="Processing full subgraphs"):
                experts = row['members']
                skills = row['required_skillset']
                for i in range(len(experts)):
                    for j in range(i + 1, len(experts)):
                        expert_pairs.append((experts[i], experts[j]))
                        expert_pairs.append((experts[j], experts[i]))
                for i in range(len(skills)):
                    for j in range(i + 1, len(skills)):
                        skill_pairs.append((skills[i], skills[j]))
                        skill_pairs.append((skills[j], skills[i]))
                for expert in experts:
                    for skill in skills:
                        expert_skill_pairs.append((expert, skill))
                        expert_skill_pairs.append((skill, expert))
            expert_pairs = list(set(expert_pairs))
            skill_pairs = list(set(skill_pairs))
            expert_skill_pairs = list(set(expert_skill_pairs))

            if expert_pairs:
                data['expert', 'connected_to', 'expert'].edge_index = torch.tensor(list(zip(*expert_pairs)), dtype=torch.long)
            if skill_pairs:
                data['skill', 'connected_to', 'skill'].edge_index = torch.tensor(list(zip(*skill_pairs)), dtype=torch.long)
            if expert_skill_pairs:
                data['expert', 'connected_to', 'skill'].edge_index = torch.tensor(list(zip(*expert_skill_pairs)), dtype=torch.long)

    data = T.ToUndirected()(data)
    torch.save(data, path)
    return data


def experts_df_from_teamsvec(teamsvec):
    # Get the non-zero indices for members and skills
    member_nonzero = teamsvec['member'].nonzero()
    skill_nonzero = teamsvec['skill'].nonzero()

    # Create a dictionary to map each expert to their skills
    expert_to_skills = {i: set() for i in range(teamsvec['member'].shape[1])}

    # Iterate over the non-zero member indices with progress tracking
    for member_row, expert_index in tqdm(zip(member_nonzero[0], member_nonzero[1]), total=len(member_nonzero[0]),
                                         desc="Processing experts"):
        # Find the skills for the given member
        skill_indices = skill_nonzero[1][skill_nonzero[0] == member_row]
        # Add the skills to the corresponding expert
        expert_to_skills[expert_index].update(skill_indices)

    # Convert the dictionary to a list of lists
    experts_list = [list(skills) for skills in expert_to_skills.values()]

    # Create the DataFrame
    experts_df = pd.DataFrame({'user_id': list(expert_to_skills.keys()), 'skills': experts_list})
    return experts_df


# def experts_df_from_teamsvec(teamsvec):
#     experts_list = []
#     for exprt in tqdm(range(teamsvec['member'].shape[1]), desc="Processing experts"):
#         experts_list.append(list(set(teamsvec['skill'][teamsvec['member'][:,exprt].nonzero()[0],:].nonzero()[1])))
#
#     experts_df = pd.DataFrame([[i, experts_list[i]] for i in range(len(experts_list))], columns=["user_id", "skills"])
#     return experts_df

def teams_df_from_teamsvec(teamsvec):
    # Get the non-zero indices for skills, members, and locations
    skill_nonzero = teamsvec['skill'].nonzero()
    member_nonzero = teamsvec['member'].nonzero()
    if 'loc' in teamsvec: location_nonzero = teamsvec['loc'].nonzero()

    # Create dictionaries to map each team to its skills, members, and location
    team_to_skills = {i: [] for i in range(teamsvec['id'].shape[0])}
    team_to_members = {i: [] for i in range(teamsvec['id'].shape[0])}
    if 'loc' in teamsvec: team_to_location = {i: [] for i in range(teamsvec['id'].shape[0])}

    # Populate the dictionaries
    for skill_row, skill_col in tqdm(zip(skill_nonzero[0], skill_nonzero[1]), total=len(skill_nonzero[0]),
                                     desc="Processing skills"):
        team_to_skills[skill_row].append(skill_col)

    for member_row, member_col in tqdm(zip(member_nonzero[0], member_nonzero[1]), total=len(member_nonzero[0]),
                                       desc="Processing members"):
        team_to_members[member_row].append(member_col)

    for loc_row, loc_col in tqdm(zip(location_nonzero[0], location_nonzero[1]), total=len(location_nonzero[0]),
                                 desc="Processing locations"):
        team_to_location[loc_row].append(loc_col)

    # Create a list of lists for DataFrame creation
    teams_sorted = [[team_id, team_to_skills[team_id], team_to_members[team_id], team_to_location[team_id]] for team_id
                    in range(teamsvec['id'].shape[0])]

    # Create the DataFrame
    teams_sorted_df = pd.DataFrame(teams_sorted, columns=['team_id', 'required_skills', 'members', 'location'])
    return teams_sorted_df

# def teams_df_from_teamsvec(teamsvec):
#     teams_sorted = []
#     for row in tqdm(range(teamsvec['id'].shape[0]), desc="Processing teams"):
#         team_id = row
#         required_skills = teamsvec['skill'][row].nonzero()[1]
#         members = teamsvec['member'][row].nonzero()[1]
#         location = teamsvec['loc'][row].nonzero()[1]
#         teams_sorted.append([team_id, required_skills, members, location])
#
#     teams_sorted_df = pd.DataFrame(teams_sorted, columns=['team_id', 'required_skills', 'members', 'location'])
#     return teams_sorted_df

# def teams_df_from_teamsvec(teamsvec):
#     teams_sorted = []
#     for row in tqdm(range(teamsvec['id'].shape[0]), desc="Processing teams"):
#         teams_sorted.append([row, teamsvec['skill'][row].nonzero()[1], teamsvec['member'][row].nonzero()[1]])
#
#     teams_sorted_df = pd.DataFrame(teams_sorted, columns=['team_id', 'required_skills', 'members'])
#     return teams_sorted_df




