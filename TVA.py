import csv
import pandas as pd
import string
import numpy as np

df = pd.read_csv('voting_example3.csv', sep=";")
CONSIDERED_VOTE = 8  # same number as candidates for BORDA, less for other types, Singularity = 1
#  Vote for two = 2, Anti plurality = range(1,n_pref)
BULLET = False
BURYING = True
STRATEGIC_VOTING = set()

class Agent(object):
    def __init__(self, n_pref, n_voters, considered_vote):
        self.n_pref = n_pref
        self.n_voters = n_voters
        self.ns_outcome = {}
        self.happiness = []
        self.winner_prefs=1

        if considered_vote > n_pref or considered_vote <= 0:  # solve basics cases
            considered_vote = n_pref
        self.considered_vote = considered_vote
        if considered_vote!=1 and considered_vote!=n_pref:
            self.winner_prefs = considered_vote


    """calculate winner given votes"""

    def calculate_score(self, arr, initial=False):  # votes is the
        votes = dict(zip(string.ascii_uppercase, [0] * self.n_pref))  # create dict to calculate outcomes from array
        weighted = False  # if type of vote is borda, we weigth the votes, else a vote is equal to 1
        if self.considered_vote == self.n_pref:
            weighted = True
        for j in range(self.considered_vote):
            for i in range(arr.shape[0]):
                # print("ind", i+1, arr[i, j])
                vote = arr[i, j]
                if vote == "-":  # no weight, bullet case
                    break
                if weighted:
                    votes[vote] += (self.n_pref - j - 1)
                else:
                    votes[vote] += 1
        outcome = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        if initial:
            self.ns_outcome = outcome
        return outcome

    """ calculate and evaluate new outcome from strategic voting"""
    def calculate_new_strategic(self, new_pref, method, voter):
        # print("new_pref", voter , new_pref)
        strategic_outcome = self.calculate_score(new_pref)
        print("##### new outcome with,",method," #### \n", strategic_outcome)
        distance = self.calculate_distance(new_pref, strategic_outcome)  # todo #distance considering the '-'?
        happiness = self.calculate_happiness(distance)
        if happiness[voter] > self.happiness[voter]:  # fixme create hash from set
            print("old value and new", happiness[voter], self.happiness[voter])
            STRATEGIC_VOTING.add(str(method)+str(voter))
            print(method,"happiness voter", voter, "\n old", self.happiness, " \n new",
                  happiness)

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
                    for j in range(self.winner_prefs, self.n_pref-1):
                        new_pref = arr.copy()
                        if new_pref[i,j] == only_pref[0]:
                            #try to lower the winner vote and calculate everything again
                            for next in (j+1,self.n_pref-1):
                                new_pref = arr.copy()
                                print(new_pref[i])
                                temp = new_pref[i,j]
                                print(next)
                                new_pref[i,j] = new_pref[i,next]
                                new_pref[i, next] = temp
                                print(new_pref[i],"voter",i, " swap",j, next,"\n\n\n")
                                self.calculate_new_strategic(new_pref, "BURYING", i)


        print(STRATEGIC_VOTING)
        return len(STRATEGIC_VOTING)

        #### NO DIFFERENCE WITH VOTING TYPE ####

    """calculate the distance necessary to calculate the happiness"""

    def calculate_distance(self, arr, sorted_outcome):  # same for every type of vote
        distance = {}
        for i in range(self.n_voters):
            # print("voter", i+1)
            max_d = self.n_pref
            distance_voter = 0
            # for (info, expressed_vote) in df.iloc[i, 1:].iteritems():
            for expressed_vote in range(arr.shape[1]):
                # print("order",expressed_vote)
                j = max_d  # J = W
                for index2, v in enumerate(sorted_outcome):
                    k = self.n_pref - index2  # calculate k from outcome
                    if v[0] == arr[i, expressed_vote]:  # look for the vote,
                        distance_i = k - j
                        distance_voter += distance_i * j  # sum of distances of all alternatives * weights:
                        break
                max_d -= 1  # decrease the j while iterate over alternative
            distance.setdefault(i, distance_voter)
        return distance

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


def main():
    print(df.shape)
    n_pref = df.shape[1] - 1  # column number without considering first number
    n_voters = df.shape[0]
    weighted_vote = True
    arr = df.to_numpy()[:, 1:]

    TVA = Agent(n_pref, n_voters, CONSIDERED_VOTE)

    ##### NON STRATEGIC VOTING OUTCOME ######
    ns_outcome = TVA.calculate_score(arr, True)

    ##### HAPPINESS #####
    distance = TVA.calculate_distance(arr, ns_outcome)
    # for d in distance:
    happiness = TVA.calculate_happiness(distance, True)

    print("##### Non strategic results""")
    print("distance", distance)
    print("outcome", ns_outcome)
    print("election winner is", ns_outcome[0])
    print("happiness", happiness)

    print("#### strategic results ####""")
    ###### STRATEGIC VOTING #####
    n_set = TVA.strategic_voting_bullet(arr)

    ###### OVERALL RISK OF SV ######

    risk = TVA.overall_risk(n_set)
    print("the risk for this situation is:", risk)

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
#  - risk
