from rdkit import Chem
import json 
from operator import methodcaller, itemgetter
from functools import partial
import numpy as np

from freedpp.env.state import State
from freedpp.utils import dmap, lmap, dsuf
from freedpp.env.utils import connect_mols

class Environment(object):
    def __init__(self, *, atom_vocab, bond_vocab, frag_vocab, timelimit=4,
                 rewards, scaffolds_json: str = None,  
                 action_size=40, fragmentation='crem', **kwargs):
        self.atom_vocab = atom_vocab
        self.frag_vocab = frag_vocab
        self.bond_vocab = bond_vocab

        if scaffolds_json is None:
            raise ValueError("Way to json (scaffolds_json).")

        with open(scaffolds_json, 'r') as f:
            scaffolds_data = json.load(f)
        
        if not isinstance(scaffolds_data, list):
            raise ValueError("JSON hasn't scaffolds.")
        
        self.starting_scaffolds = scaffolds_data

        assert fragmentation in ['crem', 'brics']
        self.fragmentation = fragmentation
        if fragmentation == 'crem':
            attach_vocab = ['*']
        elif fragmentation == 'brics':
            attach_vocab = [f"[{i}*]" for i in range(1, 17)]
            attach_vocab.remove("[2*]")
 
        self.attach_vocab = attach_vocab
        self.num_att_types = len(attach_vocab)
        self.atom_dim = len(atom_vocab) + len(attach_vocab) + 18
        self.bond_dim = len(self.bond_vocab)
        self.action_size = action_size
        self.state_args = {
            'fragmentation': fragmentation,
            'atom_dim': self.atom_dim,
            'bond_dim': self.bond_dim,
            'atom_vocab': atom_vocab,
            'bond_vocab': bond_vocab,
            'attach_vocab': attach_vocab
        }
        self.num_steps = 0
        self.state = None 
        self.rewards = rewards
        self.timelimit = timelimit
        self.fragments = [State(frag, 0, **self.state_args) for frag in self.frag_vocab]

        num_att = [len(frag.attachments) for frag in self.fragments]
        N, M = len(self.frag_vocab), max(num_att)
        S = M  
        T = self.timelimit
        self.actions_dim = (S + T * (M - 1), N, M)

    def reset(self, scaffold_idx: int = None):
        self.num_steps = 0
        
        if scaffold_idx is None:
            start_smile = np.random.choice(self.starting_scaffolds)
        else:
            start_smile = self.starting_scaffolds[scaffold_idx]
            
        self.state = State(start_smile, self.num_steps, **self.state_args)
        
        num_att = [len(frag.attachments) for frag in self.fragments]
        S, T = len(self.state.attachments), self.timelimit
        N, M = len(self.frag_vocab), max(num_att)
        self.actions_dim = (S + T * (M - 1), N, M)  # Пересчитать actions_dim после выбора скаффолда
        return self.state

    def reward_batch(self, smiles):
        objectives = dmap(methodcaller('__call__', smiles), self.rewards)
        rewards = dsuf('Reward', dmap(partial(lmap, itemgetter(0)), objectives))
        properties = dsuf('Property', dmap(partial(lmap, itemgetter(1)), objectives))
        rewards['Reward'] = np.sum(list(rewards.values()), axis=0).tolist()
        return {**rewards, **properties}

    def step(self, action):
        self.attach_fragment(action)
        self.num_steps += 1
        terminated = not self.state.attachments
        truncated = self.num_steps >= self.timelimit
        # reward calculated only for terminal states in "reward_batch" call
        reward = 0.
        state = self.state
        info = dict()
        return state, reward, terminated, truncated, info

    def attach_fragment(self, action):
        a1, a2, a3 = action
        mol = self.state.molecule
        frag_state = self.fragments[a2]
        frag = frag_state.molecule
        mol_attachments = self.state.attachment_ids
        mol_attachment = mol.GetAtomWithIdx(mol_attachments[a1])
        frag_attachments = frag_state.attachment_ids
        frag_attachment = frag.GetAtomWithIdx(frag_attachments[a3])
        new_mol = connect_mols(mol, frag, mol_attachment, frag_attachment)
        self.state = State(Chem.MolToSmiles(new_mol), self.num_steps + 1, **self.state_args)