import csv
import pandas as pd
import string
import numpy as np

df = pd.read_csv('voting_example3.csv', sep=";")


class Agent(object):
    def __init__(self, n_pref, n_voters):
        self.n_pref = n_pref
        self.n_voters = n_voters
        self.ns_outcome = {}

    """calculate winner given votes"""
    def calculate_score(self, votes, initial = False):
        for i in range(self.n_pref):
            for (index, j) in df.iloc[:, i + 1].iteritems():
                votes[j] += (self.n_pref - i - 1)
        outcome = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        if initial:
            self.ns_outcome = outcome
        return outcome

    def calculate_distance(self, sorted_outcome):
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

    """ calculate the happiness of the single voter given his distance"""

    def calculate_happiness(self, d):
        # print("happiness",d,1 / (1 + np.abs(d)))
        return 1 / (1 + np.abs(d))

    def overall_risk(self, S):
        return S/self.n_voters


def main():
    print(df.shape)
    n_pref = df.shape[1] - 1
    n_voters = df.shape[0]
    votes = dict(zip(string.ascii_uppercase, [0] * n_pref))
    happiness = []

    TVA = Agent(n_pref, n_voters)

    ##### NON STRATEGIC VOTING OUTCOME ######
    ns_outcome = TVA.calculate_score(votes, True)

    ##### HAPPINESS #####
    distance = TVA.calculate_distance(ns_outcome)
    for d in distance:
        happiness.append(TVA.calculate_happiness(distance[d]))

    print("##### Non strategic results""")
    print("distance", distance)
    print("outcome", ns_outcome)
    print("election winner is", ns_outcome[0])
    print("happiness", happiness)

    print("#### strategic results ####""")
    ###### STRATEGIC VOTING #####
    strategic_voting = {}

    # voters = voters[0]
    for i in range(n_voters):
        voter = list(df.iloc[i,1:])
        max_d = n_pref
        j = 0
        for pos,vote in enumerate(voter):
            print(pos, vote)
            if pos == 0 and vote == ns_outcome[0][0]:
                # this voter choice is already the winner
                print("this voter choice is already the winner")
                break

    ###### OVERALL RISK OF SV ######
    #FIXME adjust when S will be available -> remove +1
    s=len(strategic_voting)+1
    risk = TVA.overall_risk(s)
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
