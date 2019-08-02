# 计算基于某种策略的状态价值
import numpy as np

class State:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    # def getState(self):
    #     return self.x,self.y
    #
    # def setState(self,x,y):
    #     self.x = x
    #     self.y = y

class Act:
    def __init__(self,dx,dy):
        self.dx = dx
        self.dy = dy

class Policy:
    def getNextActWithNetwork(self,stat):
        return networkGetAct(stat)
    def getNextAct(self,stat):
        if (stat.x==0 and stat.y==0) or (stat.x==3 and stat.y==3):
            return Act(0,0)
        else:
            rd = np.random.random()
            if rd<0.25:
                return Act(-1,0)
            elif rd<0.5:
                return Act(1,0)
            elif rd<0.75:
                return Act(0,-1)
            else:
                return Act(0,1)
    def getNextActProb(self,stat):
        if (stat.x==0 and stat.y==0) or (stat.x==3 and stat.y==3):
            return {Act(0,0):1,Act(-1,0):0,Act(1,0):0,Act(0,-1):0,Act(0,1):0}
        else:
            return {Act(0,0):0,Act(-1,0):0.25,Act(1,0):0.25,Act(0,-1):0.25,Act(0,1):0.25}

class Rule:                                                                 # describe state distribution after stat-act
    def getNextStateWithActWithNetwork(self,stat,act):
        return networkGetState(stat,act)
    def getNextStateWithAct(self,stat,act):
        if stat.x==0 and act.dx==-1:
            return stat
        elif stat.x==3 and act.dx==1:
            return stat
        elif stat.y == 0 and act.dy == -1:
            return stat
        elif stat.y==3 and act.dy==1:
            return stat
        else:
            return State(stat.x+act.dx, stat.y+act.dy)
###############################################################################################
gama = 1
actSet = (Act(-1,0),Act(1,0),Act(0,-1),Act(0,1))
actSetEx = (Act(0,0),)
statSet = set()
for x in range(0,4):
    for y in range(0,4):
        statSet.add(State(x,y))
rule = Rule()
policy = Policy()
# policy = "random"
statAfterAction = "fix"                                                     # or a dict represent stat probability distribution
##########################################################################################
class StatActionCurScore:
    def __init__(self):
        self.statActionCurScoreDict = {}
        for x in range(0, 4):
            for y in range(0, 4):
                if (x == 0 and y == 0) or (x == 3 and y == 3):
                    for act in actSetEx:
                        self.statActionCurScoreDict[(State(x, y), act)] = 0
                else:
                    for act in actSet:
                        self.statActionCurScoreDict[(State(x, y), act)] = -1
    def getScore(self,stat,act):
        for sta_cat in self.statActionCurScoreDict.keys():
            if stat.x == sta_cat[0].x and stat.y == sta_cat[0].y and act.dx == sta_cat[1].dx and act.dy == sta_cat[1].dy:
                return self.statActionCurScoreDict[sta_cat]
        return None

statActionCurScore = StatActionCurScore()
##########################################################################################

class StatScore:
    def __init__(self):
        self.statScoreDict={}
        for stat in statSet:
            self.statScoreDict[stat] = 0
    def getStatScore(self,stat):
        for sta in self.statScoreDict.keys():
            if stat.x == sta.x and stat.y == sta.y:
                return self.statScoreDict[sta]
        return None

    def iterate(self):
        for stat in statSet:
            actProbDict = policy.getNextActProb(stat)
            score = 0
            for act in actProbDict.keys():
                curActScore = statActionCurScore.getScore(stat,act)
                if curActScore is None:
                    continue
                nextStat = rule.getNextStateWithAct(stat,act)
                score = score + actProbDict[act]*(curActScore+gama*self.getStatScore(nextStat))
            self.statScoreDict[stat] = score

if __name__ == "__main__":
    statScore = StatScore()
    scoreView = np.zeros([4,4],dtype=np.float32)
    for i in range(0,1000):
        statScore.iterate()
        for stat in statScore.statScoreDict.keys():
            scoreView[stat.x, stat.y] = statScore.statScoreDict[stat]
        print(scoreView)