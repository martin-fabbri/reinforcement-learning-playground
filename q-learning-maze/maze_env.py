import numpy as np


#https://github.com/openai/gym/blob/master/gym/spaces/prng.py
np_random = np.random.RandomState()
# TODO: CONTINUE ...https://github.com/openai/gym/blob/master/gym/spaces/prng.py


class Space(object):
    """Defines the observation and action spaces, so you can write generic
    code that applies to any Env. For example, you can choose a random
    action.
    """

    # TODO: CONTINUE ...https://github.com/openai/gym/blob/master/gym/spaces/prng.py
    def seed(seed=None):
        """Seed the common numpy.random.RandomState used in spaces
        CF
        https://github.com/openai/gym/commit/58e6aa95e5af2c738557431f812abb81c505a7cf#commitcomment-17669277
        for some details about why we seed the spaces separately from the
        envs, but tl;dr is that it's pretty uncommon for them to be used
        within an actual algorithm, and the code becomes simpler to just
        use this common numpy.random.RandomState.
        """
        np_random.seed(seed)

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        raise NotImplementedError

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError


class MazeEnv(object):
    """Has the following members
    - nS: Number of states
    - nA: Number of actions
    - P: transitions
    """
    def _init__(self, nS, nA, P):
        self.P = P
        self.last_action = None  # use for rendering
        self.nS = nS
        self.nA = nA

        self.action_space =