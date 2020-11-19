import csv
import pandas as pd
import string
import numpy as np
from numpy import random

############  GLOBAL PARAMS #################
PLURALITY_VOTE = 1
VOTING_FOR_2 = 2
VETO = 3
BORDA = 4
STRATEGIC_VOTING = set()


############  CONFIG #################
BULLET       = True
COMPROMISING = True
BURYING      = True
LONG_RUN     = True # if you want 100 trial of random tables, otherwise load the csv file
CONSIDERED_VOTE = BORDA



class Agent(object):

    def __init__(self, n_pref, n_voters, vote_type):
        self.n_preferences = n_pref
        self.n_voters = n_voters
        self.vote_type = vote_type
        self.ns_outcome = {}
        self.happiness = []
        self.considered_vote = self.nunmber_of_considered_votes(vote_type)
        self.winner_prefs = self.considered_vote if self.vote_type != PLURALITY_VOTE and self.vote_type != BORDA else 1


    def nunmber_of_considered_votes(self, vote_type):
        if vote_type == PLURALITY_VOTE:
            return 1
        elif vote_type == VOTING_FOR_2:
            return 2
        elif vote_type == VETO:
            return self.n_preferences - 1
        elif vote_type == BORDA:
            return self.n_preferences
        else: # something wrong but still BORDA
            print("ERROR: wrong vote_type", vote_type)
            return self.n_preferences


    """calculate winner given votes"""
    def calculate_score(self, table, initial=False):
        votes_result = dict(zip(string.ascii_uppercase, [0] * self.n_preferences))  # init dict to calculate outcomes from table

        for i in range(self.n_voters):
            for j in range(self.considered_vote):  # check only the column useful for your vote_type
                vote = table[i, j]
                if vote == "-":  # bullet case, avoid further "-" checks
                    break
                if self.vote_type == BORDA:  # if type of vote is BORDA, we weigth the votes from n-1 to 0, else a vote is equal to 1
                    votes_result[vote] += (self.n_preferences - j - 1)
                else:
                    votes_result[vote] += 1

        outcome = sorted(votes_result.items(), key=lambda x: x[1], reverse=True)

        if initial:
            self.ns_outcome = outcome

        return outcome


    """calculate the distance necessary to calcualte the happiness"""
    def calculate_distance(self, table, outcome):  # same for every type of vote
        distance = {}
        for i in range(self.n_voters):
            # print("voter", i+1)
            max_d = self.n_preferences
            distance_voter = 0
            # for (info, expressed_vote) in df.iloc[i, 1:].iteritems():
            for expressed_vote in range(self.n_preferences):
                # print("order",expressed_vote)
                j = max_d  # J = W
                for index2, v in enumerate(outcome):
                    k = self.n_preferences - index2  # calculate k from outcome
                    if v[0] == table[i, expressed_vote]:  # look for the vote,
                        distance_i = k - j
                        distance_voter += distance_i * j  # sum of distances of all alternatives * weights:
                        break
                max_d -= 1  # decrease the j while iterate over alternative
            distance.setdefault(i, distance_voter)
        return distance


    """ calculate the happiness of the single voter given his distance"""
    def calculate_happiness(self, distance, initial=False):
        # print("happiness",d,1 / (1 + np.abs(d)))
        happiness = []

        for d in distance:
            dist_value = distance[d]
            happiness.append(1 / (1 + np.abs(dist_value)))

        if initial == True:
            self.happiness = happiness

        return happiness


    def overall_risk(self, S):
        return S / self.n_voters


    """ calculate and evaluate new outcome from strategic voting"""
    def calculate_new_strategic(self, new_pref, method, voter):
        # print("new_pref", voter , new_pref)
        strategic_outcome = self.calculate_score(new_pref)
        print("##### new outcome with,", method, " #### \n", strategic_outcome)
        distance = self.calculate_distance(new_pref, strategic_outcome)  # todo #distance considering the '-'?
        happiness = self.calculate_happiness(distance)
        if happiness[voter] > self.happiness[voter]:  # fixme create hash from set
            print("old value and new", happiness[voter], self.happiness[voter])
            STRATEGIC_VOTING.add(str(method)+str(voter))
            print(method, "happiness voter", voter, "\n old", self.happiness, " \n new", happiness)


    """ calculate the happiness of the single voter given his distance"""
    # not working, do not consider it
    def strategic_voting_bullet(self, arr): #todo consider winner case
        # new_pref = arr.copy()
        only_pref = list(list(zip(*self.ns_outcome))[0])  # make a list with only the preferences, no score
        # print("real",real_outcome)

        if BULLET and CONSIDERED_VOTE > 1:
            for i in range(self.n_voters):
                new_pref = arr.copy()
                if new_pref[i, 0] == only_pref[0]:  # skip if the first choice is already the winner
                                                    #todo consider more than just 1 pref?
                    pass
                else:
                    # for now just exclude the other votes except the first one, no swap votes
                    print("######## BULLET VOTING ########")  # don't consider when my second chance can win

                    for j in range(1, arr.shape[1]):  # set all the other preferences to a null value
                        new_pref[i, j] = '-'
                    # print(new_pref)
                    self.calculate_new_strategic(new_pref,"BULLET",i)

        if BURYING: #fixme work in progress
            print("######## BURYING VOTING ########")
            for i in range(self.n_voters):
                new_pref = arr.copy()
                winner = False
                for j in range(self.winner_prefs):
                    if new_pref[i, j] == only_pref[0]: #skip if the winner is in my choices
                        winner = True
                if not winner:
                    for j in range(self.winner_prefs, self.n_preferences-1):
                        new_pref = arr.copy()
                        if new_pref[i, j] == only_pref[0]:
                            #try to lower the winner vote and calculate everything again
                            for next in (j+1, self.n_preferences-1):
                                new_pref = arr.copy()
                                print(new_pref[i])
                                temp = new_pref[i, j]
                                print(next)
                                new_pref[i, j] = new_pref[i, next]
                                new_pref[i, next] = temp
                                print(new_pref[i], "voter", i, " swap", j, next, "\n\n\n")
                                self.calculate_new_strategic(new_pref, "BURYING", i)


        print(STRATEGIC_VOTING)
        return len(STRATEGIC_VOTING)

        #### NO DIFFERENCE WITH VOTING TYPE ####


def initialize_random_tables(number, n_voters, n_preferences):  # MAX 26 preferences
    if n_preferences > 26:  # we want use only the A-Z chars
        n_preferences = 26
    elif n_preferences < 1:
        n_preferences = 1

    random.seed(1)

    table_list = []
    for n in range(number):
        voters_table = []
        for v in range(n_voters):
            random_row = random.permutation(n_preferences) + 65  # for the ASCII code
            single_voter_row = np.array([chr(value) for value in random_row])
            voters_table.append(single_voter_row)
        table_list.append(np.array(voters_table))
    return np.array(table_list)

def main():
    if LONG_RUN:
        df = initialize_random_tables(100, 6, 5)
    else:
        df = pd.read_csv('voting_example3.csv', sep=";").to_numpy()[:, 1:] # not considering the first column of voters's IDs
        df = np.expand_dims(df, axis=0)

    n_test, n_voters, n_preferences = df.shape

    for n, table in enumerate(df):
        print("\n\n##########  TEST ", n, "   ##########")
        TVA = Agent(n_preferences, n_voters, CONSIDERED_VOTE)

        ##### NON STRATEGIC VOTING OUTCOME ######
        ns_outcome = TVA.calculate_score(table, True)

        #####  DISTANCE & HAPPINESS  #####
        distance = TVA.calculate_distance(table, ns_outcome)
        happiness = TVA.calculate_happiness(distance, True)

        print("\n#####  NON STRATEGIC RESULTS  #####""")
        print("distance:        ", distance)
        print("outcome:         ", ns_outcome)
        print("election winner: ", ns_outcome[0])
        print("happiness:       ", happiness)



        print("\n####  STRATEGIC RESULTS  ####""")
        ###### STRATEGIC VOTING #####
        n_set = TVA.strategic_voting_bullet(table)
        ###### OVERALL RISK OF SV ######
        risk = TVA.overall_risk(n_set)
        print("the risk for this situation is:", risk)


if __name__ == "__main__":
    main()

"""
expected OUTPUT
- non strategic voting outcome O
- Overall voter happiness
- Possibly empty set of strategic voting option
   - v new list of strategic voting option,
   - O voting results applying v
   - H resulting happiness
- Risk
"""

