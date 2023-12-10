
import torch


class Data_Sampler(object):
    def __init__(self, data, device, start=None, num=1000):
        if start is None:

            self.state = torch.from_numpy(data['observations']).float()[:num]
            self.action = torch.from_numpy(data['actions']).float()[:num]

        else:
            self.state = torch.from_numpy(data['observations']).float()[start:start + num]
            self.action = torch.from_numpy(data['actions']).float()[start:start + num]

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        self.device = device
        self.length = num

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
        )

    def order_sample(self, start, batch_size):
        if start + batch_size > self.length:
            return (
                self.state[start:].to(self.device),
                self.action[start:].to(self.device),
            )
        else:
            return (
                self.state[start:start + batch_size].to(self.device),
                self.action[start:start + batch_size].to(self.device),
            )
