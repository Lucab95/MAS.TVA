import csv
import pandas as pd
import string
import numpy as np

df = pd.read_csv('voting_example3.csv', sep=";")
CONSIDERED_VOTE = 6  #  same number as candidates for BORDA, less for other types, Singularity = 1
                     #  Vote for two = 2, Anti plurality = range(1,n_pref)


class Agent(object):
    def __init__(self, n_pref, n_voters,considered_vote):
        self.n_pref = n_pref
        self.n_voters = n_voters
        self.ns_outcome = {}

        if considered_vote>n_pref or considered_vote <= 0:  # solve basics cases
            considered_vote=n_pref
        self.considered_vote = considered_vote


    """calculate winner given votes"""
    def calculate_score(self, arr, initial = False): #votes is the
        votes = dict(zip(string.ascii_uppercase, [0] * self.n_pref)) #create dict to calculate outcomes from array
        weighted = False    # if type of vote is borda, we weigth the votes, else a vote is equal to 1
        if self.considered_vote == self.n_pref:
            weighted = True
        for j in range(self.considered_vote):
            for i in range(arr.shape[0]):
                print("ind", i+1, arr[i, j])
                vote = arr[i, j]
                if weighted:
                    votes[vote] += (self.n_pref - j - 1)
                else:
                    votes[vote] += 1
        outcome = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        if initial:
            self.ns_outcome = outcome
        return outcome

    """ calculate the happiness of the single voter given his distance"""
    #not working, do not consider it
    def strategic_voting(self):
        bullet = True
        # if bullet:
        strategic_voting = {}
        for i in range(self.n_voters):
            voter = list(df.iloc[i, 1:])  # get the preference's voter row
            for pos, vote in enumerate(voter):
                print(pos, vote)
                if pos == 0 and vote == self.ns_outcome[0][0]:

                #     print("this voter choice is already the winner")
                    break
                # elif cases of other votes
        return len(strategic_voting)


            #### NO DIFFERENCE WITH VOTING TYPE ####
    """calculate the distance necessary to calcualte the happiness"""
    def calculate_distance(self, sorted_outcome): #same for every type of vote
        distance = {}
        for i in range(self.n_voters):
            # print("voter", i+1)
            max_d = self.n_pref
            distance_voter = 0
            for (info, expressed_vote) in df.iloc[i, 1:].iteritems():
                j = max_d  # J = W
                for index2, v in enumerate(sorted_outcome):
                    k = self.n_pref - index2  # calculate k from outcome
                    if v[0] == expressed_vote:  # look for the vote #TODO can be improved
                        distance_i = k - j
                        distance_voter += distance_i * j  # sum of distances of all alternatives * weights:
                        break
                max_d -= 1  # decrease the j while iterate over alternative
            distance.setdefault(i, distance_voter)
        return distance

    def calculate_happiness(self, d):
        # print("happiness",d,1 / (1 + np.abs(d)))
        return 1 / (1 + np.abs(d))

    def overall_risk(self, S):
        return S/self.n_voters


def main():
    print(df.shape)
    n_pref = df.shape[1] - 1 #base case is borda
    n_voters = df.shape[0]
    happiness = []
    weighted_vote = True
    arr = df.to_numpy()[:,1:]

    TVA = Agent(n_pref, n_voters, CONSIDERED_VOTE)

    ##### NON STRATEGIC VOTING OUTCOME ######
    ns_outcome = TVA.calculate_score(arr,True)

    ##### HAPPINESS #####
    distance = TVA.calculate_distance(ns_outcome)
    for d in distance:

        happ = TVA.calculate_happiness(d)
        happiness.append(happ)

    print("##### Non strategic results""")
    print("distance", distance)
    print("outcome", ns_outcome)
    print("election winner is", ns_outcome[0])
    print("happiness", happiness)

    print("#### strategic results ####""")
    ###### STRATEGIC VOTING #####
    n_set = TVA.strategic_voting()

    ###### OVERALL RISK OF SV ######

    risk = TVA.overall_risk(n_set)
    print("the risk for this situation is:",risk)

    # print(vote)


if __name__ == "__main__":
    main()

# expeceted OUTPUT
#  - non strategic voting outcome O
#  - Overall voter happiness
#  - Possibly empty set of strategic voting option
#     - v new list of strategic voting option,
#     - O voting results applying v
#     - H resulting happiness
#     -
