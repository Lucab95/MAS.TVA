import sys
import csv
import pandas as pd
import string
import numpy as np
from numpy import random
import copy, statistics, time
from itertools import permutations


############  GLOBAL PARAMS #################
PLURALITY_VOTE = 1
VOTING_FOR_2 = 2
VETO = 3
BORDA = 4


############  CONFIG #################
BULLET       = True
COMPROMISING = True
BURYING      = True
LONG_RUN     = False  # if you want 100 trial of random tables, otherwise load the csv file
#If BRUTEFORCE is active -> no possibility to decide the strategic voting used
BRUTE_FORCE = True
CONSIDERED_VOTE = BORDA
#if true add the method used to find the strategic preference
SPECIFY_METHOD_IN_SET= False
DF_NAME = 'voting_example3.csv'
SAVING_LOG = True
LOG_NAME ='log.txt'


class Agent(object):

    def __init__(self, n_pref, n_voters, vote_type,table):
        self.n_preferences = n_pref
        self.n_voters = n_voters
        self.vote_type = vote_type
        self.ns_outcome = {}
        self.ns_preferences = table
        self.happiness = []
        self.strategic_voting = set()
        self.set_compromising = set()
        self.set_burying= set()
        self.set_bullett= set()
        self.brute_force= set()
        self.considered_vote = self.number_of_considered_votes(vote_type)
        self.winner_prefs = self.considered_vote if self.vote_type != PLURALITY_VOTE and self.vote_type != BORDA else 1
        self.election_happ_incr_list=[]
        self.avg_election_happ_incr=0


    def number_of_considered_votes(self, vote_type):
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
            print("ERROR: by default is set to BORDA")
            return self.n_preferences


    """calculate winner given votes"""
    def calculate_outcome(self, table, initial=False):
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


    """calculate the distance necessary to calculate the happiness"""
    def calculate_distance(self, table, outcome):  # same for every type of vote
        distance = {}

        for i in range(self.n_voters):
            # print("voter_pref", table[i])
            # print("outcome ", np.array([x[0] for x in outcome]))

            distance_value = 0
            Wj = self.n_preferences # max value of the weights

            for j in range(self.n_preferences):
                for index2, v in enumerate(outcome):
                    k = self.n_preferences - index2  # calculate k from outcome
                    if v[0] == table[i, j]:  # look for the vote,
                        distance_j = k - Wj
                        distance_value += distance_j * Wj  # sum of distances of all alternatives * weights:
                        break

                Wj -= 1  # decrease the current_w while iterate over alternative -> Wj

            distance.setdefault(i, distance_value)

        return distance


    """ calculate the happiness of the single voter given his distance"""
    def calculate_happiness(self, distance, initial_ns_voting=False):
        # print("happiness",d,1 / (1 + np.abs(d)))
        happiness = []

        for d in distance:
            dist_value = distance[d]
            happ=round(1 / (1 + np.abs(dist_value)),5)
            happiness.append(happ)


        if initial_ns_voting == True:
            self.happiness = happiness

        return happiness


    def overall_risk(self, S):
        return S / self.n_voters


    """ calculate and evaluate new outcome from strategic voting"""
    def calculate_new_strategic(self, new_pref, method, voter):
        # print("new_pref", voter , new_pref)
        strategic_outcome = self.calculate_outcome(new_pref)
        # print("##### new outcome with", method, " #### \n", strategic_outcome)
        distance = self.calculate_distance(self.ns_preferences, strategic_outcome)
        happiness = self.calculate_happiness(distance)
        if happiness[voter] > self.happiness[voter]:
            self.election_happ_incr_list.append(happiness[voter])
            diff = happiness[voter]-self.happiness[voter]
            # print("new value and old", happiness[voter], self.happiness[voter], "diff", round(diff, 5))
            set_str = str(voter + 1)
            for i, z in enumerate(new_pref[voter]): #create a univoque string for the set
                set_str += z
            if SPECIFY_METHOD_IN_SET:
                set_str += "."+method
            self.strategic_voting.add(set_str)
            # print(method, "happiness voter", voter, "\n old", self.happiness, " \n new",
            #       happiness, "\n\n\n")
            print("voter ",voter,"outcome:",strategic_outcome,"gave an happiness increase of:",happiness[voter])
            if method =="BULLET":
                self.set_bullett.add(set_str)
            elif method =="BURYING":
                self.set_burying.add(set_str)
            elif method =="COMPROMISING":
                self.set_compromising.add(set_str)
            else:
                self.brute_force.add(set_str)
        else:
            # print("lower or same happiness")
            pass


    """ calculate the happiness of the single voter given his distance"""
    # not working, do not consider it
    def strategic_voting_types(self, table):
        # new_pref = copy.deepcopy(table)
        only_pref = list(list(zip(*self.ns_outcome))[0])  # make a list with only the preferences, no score
        winner_vote = only_pref[0]
        print("true preferences \n", self.ns_preferences)


        # working and considers all the cases, also changing my first preference
        if BULLET and CONSIDERED_VOTE > 1:
            print("######## BULLET VOTING ########")
            for i in range(self.n_voters):
                new_pref = copy.deepcopy(table)
                if new_pref[i, 0] == winner_vote:  # skip if the first choice is already the winner
                    pass
                else:
                    if BRUTE_FORCE:
                        pos_winning_pref = 1
                        for pos in range(1, len(new_pref[i])):
                            if new_pref[i][pos] == winner_vote:
                                pos_winning_pref = pos
                                break
                        # put all the preferences before the winning one on top and vote only for that one.
                        for other_vote in range(0, pos_winning_pref):
                            new_pref = copy.deepcopy(table)
                            new_pref[i, 0] = new_pref[i, other_vote]
                            # set all the other preferences to a null value
                            for j in range(1, table.shape[1]):
                                new_pref[i, j] = '-'
                            # print(new_pref)
                            self.calculate_new_strategic(new_pref,"BULLET",i)
                            # for iterator in range(0,pos_winning_pref):

                    for j in range(1, table.shape[1]):  # set all the other preferences to a null value
                        new_pref[i, j] = '-'
                    # print(new_pref)
                    self.calculate_new_strategic(new_pref,"BULLET",i)


        if not BRUTE_FORCE:
            # the logic here is that we don't change our preferences.
            #fixme Consider all the preferences-> done
            if BURYING:
                #slide the winner vote to all the next pos and for each calculate the outcome
                print("######## BURYING VOTING ########")
                for i in range(self.n_voters):
                    new_pref = copy.deepcopy(table)
                    winner = False
                    for j in range(self.winner_prefs):
                        # skip if the winner is in my n choices -> related to the voting scheme
                        if new_pref[i, j] == winner_vote:
                            winner = True
                    if not winner:
                        # pos_winning_pref = self.winner_prefs
                        # new_pref = copy.deepcopy(table)  # do another copy to avoid problem while trying diff pos
                        # for pos in range(self.winner_prefs, len(new_pref[i])):  #look for the winning vote position
                        #     if new_pref[i][pos] == winner_vote:
                        #         pos_winning_pref = pos
                        #         # print("length", len(new_pref[i]),pos_winning_pref)
                        #         break
                        for z in range(len(new_pref[i])):
                            new_pref = copy.deepcopy(table)
                            for cur_win_pos in range(z,len(new_pref[i])-1):
                                next = cur_win_pos+1
                                # print("entra?",cur_win_pos,new_pref[i][cur_win_pos],new_pref[i][next])
                                # print("old_pref", new_pref[i])
                                temp = new_pref[i, cur_win_pos]
                                new_pref[i, cur_win_pos] = new_pref[i, next]
                                new_pref[i, next] = temp
                                # print("new_pref", new_pref[i])

                                self.calculate_new_strategic(new_pref, "BURYING", i)

            if COMPROMISING:
                #slide the winner vote to all the prev pos and for each calculate the outcome
                print("######## COMPROSING VOTING ########")
                for i in range(self.n_voters):
                    new_pref = copy.deepcopy(table)
                    winner = False
                    for j in range(self.winner_prefs):
                        # skip if the winner is in my n choices -> related to the voting scheme
                        if new_pref[i, j] == winner_vote:
                            winner = True
                    if not winner:
                        # pos_winning_pref = self.winner_prefs
                        # new_pref = copy.deepcopy(table)  # do another copy to avoid problem while trying diff pos
                        # for pos in range(self.winner_prefs, len(new_pref[i])):  #look for the winning vote position
                        #     if new_pref[i][pos] == winner_vote:
                        #         # print("length", len(new_pref[i]),pos_winning_pref)
                        #         break
                        for z in range(len(new_pref[i])-1,0,-1):
                            new_pref = copy.deepcopy(table)
                            for cur_win_pos in range(z, 0, -1):
                                # if cur_win_pos==pos:
                                #     print("voter pass",i,cur_win_pos)
                                #     pass
                                # else:
                                #     print("voter not pass", i, cur_win_pos)
                                prev = cur_win_pos-1
                                # print("entra?",cur_win_pos,new_pref[i][cur_win_pos],new_pref[i][next])
                                # print("old_pref", new_pref[i])
                                temp = new_pref[i, cur_win_pos]
                                new_pref[i, cur_win_pos] = new_pref[i, prev]
                                new_pref[i, prev] = temp
                                # print("new_pref",new_pref[i])
                                self.calculate_new_strategic(new_pref, "COMPROMISING", i)
        else:
            print("######## BRUTEFORCE VOTING CALCULATION ########")

            for i in range(self.n_voters):
                    new_pref = copy.deepcopy(self.ns_preferences)
                    perms = permutations(new_pref[i])
                    for voter_perm in list(perms):
                        #print(" ",voter_perm)
                        new_pref = copy.deepcopy(self.ns_preferences)
                        new_pref[i]=voter_perm
                        self.calculate_new_strategic(new_pref, "BruteForce", i)

        print("We have a total of ", len(self.strategic_voting), "differents strategic voting")
        if len(self.strategic_voting)>0:
            print("The options are:\n", self.strategic_voting)
            # if len(self.election_happ_incr_list)>0:
            self.avg_election_happ_incr = statistics.mean(self.election_happ_incr_list)
            print("The averages increase for this election, using strategic voting is",self.avg_election_happ_incr)

        return len(self.strategic_voting)

        #### NO DIFFERENCE WITH VOTING TYPE ####



def initialize_random_tables(number, n_voters, n_preferences):  # MAX 26 preferences
    if n_preferences > 26:  # we use only the A-Z chars
        n_preferences = 26
    elif n_preferences < 1: # at least 1
        n_preferences = 1

    # random.seed(2)

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
    start_time = time.time()
    avg_risk=0
    avg_time = []
    elections_happiness = 0
    election_strategic_happiness_incr=0

    if LONG_RUN:
        df = initialize_random_tables(100, 5, 5)
    else:
        df = pd.read_csv(DF_NAME, sep=";").to_numpy()[:, 1:] # not considering the first column of voters's IDs
        df = np.expand_dims(df, axis=0)

    n_test, n_voters, n_preferences = df.shape

    # CHECK DF if it is well-formed
    error = False
    for n, table in enumerate(df):
        check_test = list(string.ascii_uppercase)[:n_preferences]
        for i, row in enumerate(table):
            for j, val in enumerate(row):
                if not val in check_test:
                    error = True
                    print("ERROR: Nan value in table(",(n+1),"), row(",(i+1),") and element position(",(j+1),")")
            if len(row) != len(set(row)):
                error = True
                print("ERROR: Double value in table(", (n + 1), "), row(", (i + 1), ")")

    if error: # to print every error in one time
        return 1

    for n, table in enumerate(df):
        exec_time = time.time()
        print("\n\n##########  TEST ", n, "   ##########")
        TVA = Agent(n_preferences, n_voters, CONSIDERED_VOTE,table)

        ##### NON STRATEGIC VOTING OUTCOME ######
        non_strategic_outcome = TVA.calculate_outcome(table, True)

        #####  DISTANCE & HAPPINESS  #####
        distance = TVA.calculate_distance(table, non_strategic_outcome)
        happiness = TVA.calculate_happiness(distance, True)
        elections_happiness += statistics.mean(happiness)
        print("\n#####  NON STRATEGIC RESULTS  #####""")
        print("distance:        ", distance)
        print("outcome:         ", non_strategic_outcome)
        print("election winner: ", non_strategic_outcome[0])
        print("happiness:       ", happiness)
        print("The non strategic happiness avg of this election is ", statistics.mean(happiness))


        print("\n####  STRATEGIC RESULTS  ####""")
        ###### STRATEGIC VOTING #####
        n_set = TVA.strategic_voting_types(table)
        if not BRUTE_FORCE:
            if BULLET:
                print("we have", len(TVA.set_bullett), "strategic voting for the Bullet method")
            if BURYING:
                print("we have", len(TVA.set_burying), "strategic voting for the Burying method")
                # print(TVA.set_burying)
            if COMPROMISING:
                print("we have", len(TVA.set_compromising), "strategic voting for the Compromising method")
                # print(TVA.set_compromising)
            if BURYING and COMPROMISING:
                intersect = TVA.set_burying.intersection(TVA.set_compromising)
                # print(intersect)
                if len(intersect)!=0:
                    print("cases in common between Burying and Compromising:",intersect)


        ###### OVERALL RISK OF SV ######
        risk = TVA.overall_risk(n_set)
        print("The risk for this situation is:", risk)
        avg_risk += risk
        avg_time.append(time.time() - exec_time)
        election_strategic_happiness_incr += TVA.avg_election_happ_incr
    if LONG_RUN:

        print("\n#### AVG ####")
        print("The average risk through all the elections is:", round(avg_risk/len(df), 4))
        print("The average of non strategic happiness through all the election is:", round(elections_happiness/len(df), 4))
        print("The average of strategic happiness increase through all the election is:", round(election_strategic_happiness_incr/len(df), 4))
        print("the average execution time is:",round(statistics.mean(avg_time),5))
    end_time = time.time() - start_time
    print("The execution took", round(end_time,5),"seconds")




if __name__ == "__main__":
    """
    expected OUTPUT
    - non strategic voting outcome O
    - Overall voter happiness
    - Possibly empty set of strategic voting option
       - v new list of strategic voting option,
       - O voting results applying v
       - H resulting happiness
       - z briefly motivation
    - Risk
    """
    if SAVING_LOG:
        orig_stdout = sys.stdout
        f = open(LOG_NAME, 'w')
        sys.stdout = f

    main()

    if SAVING_LOG:
        sys.stdout = orig_stdout
        f.close()

#todo consider initial preferences while calculating happiness -> done
#todo avg of tests
#todo save results on file
