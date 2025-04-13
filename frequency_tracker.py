import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn as nn

class FrequencyTracker:

    # New methods to track rotary embedding learnable parameters

    def __init__(self, model):
        self.model = model
        self.initial_freqs = {}
        self.initial_params = {}  # Store other learnable parameters (theta, rot_scale, rot_count)
        self.history = defaultdict(list)
        self.steps = []
        
        # Store initial values from all rotary embeddings
        for name, module in model.named_modules():
            # Check if module has the characteristics of a RotaryEmbedding
            if hasattr(module, 'freqs') and hasattr(module, 'rotate_'):
                self.initial_freqs[name] = module.freqs.clone().detach().cpu()
                
                # Track additional learnable parameters if they exist
                param_dict = {}
                # if hasattr(module, 'theta_param'):
                #     param_dict['theta'] = module.theta_param.clone().detach().cpu()
                if hasattr(module, 'rscale') and isinstance(module.rscale, nn.Parameter):
                    param_dict['rscale'] = module.rscale.clone().detach().cpu()
                if hasattr(module, 'rot') and isinstance(module.rot, nn.Parameter):
                    param_dict['rot'] = module.rot.clone().detach().cpu()
                
                if param_dict:  # Only store if we found parameters
                    self.initial_params[name] = param_dict
                    
                print(f"Found rotary embedding: {name}")

    def check_changes(self, step=None, verbose=False):
        """Check changes in frequencies and other learnable parameters, store in history"""
        changes = {}
        
        for name, module in self.model.named_modules():
            # Check if module has the characteristics of a RotaryEmbedding
            if hasattr(module, 'freqs') and name in self.initial_freqs:
                # Track frequency changes (existing code)
                current = module.freqs.clone().detach().cpu()
                initial = self.initial_freqs[name]
                
                # Calculate relative change
                change = torch.abs((current - initial) / (initial + 1e-7))
                avg_change = change.mean().item() * 100
                max_change = change.max().item() * 100
                
                # Store current frequencies and changes
                self.history[f"{name}_current"].append(current.numpy())
                self.history[f"{name}_avg_change"].append(avg_change)
                self.history[f"{name}_max_change"].append(max_change)
                
                changes[name] = {
                    "avg_change": avg_change,
                    "max_change": max_change,
                    "current": current
                }
                
                # Track other learnable parameters
                if name in self.initial_params:
                    for param_name, initial_val in self.initial_params[name].items():
                        if hasattr(module, param_name) and isinstance(getattr(module, param_name), nn.Parameter):
                            current_val = getattr(module, param_name).clone().detach().cpu()
                            # Calculate relative change
                            param_change = 100 * torch.abs((current_val - initial_val) / (initial_val + 1e-7))
                            param_change_val = param_change.mean().item()
                            
                            # Store values
                            self.history[f"{name}_{param_name}"].append(current_val.numpy())
                            self.history[f"{name}_{param_name}_change"].append(param_change_val)
                            
                            # Add to changes dict
                            changes.setdefault(name, {}).update({
                                f"{param_name}": current_val.item(),
                                f"{param_name}_change": param_change_val
                            })
                        elif hasattr(module, param_name):
                            # Handle non-parameter attributes (e.g., if they were converted to params)
                            current_val = torch.tensor(getattr(module, param_name))
                            self.history[f"{name}_{param_name}"].append(current_val.item())

                if verbose:
                    print(f"{name}: Avg freq change: {avg_change:.2f}%, Max change: {max_change:.2f}%")
                    # Print parameter changes
                    if name in self.initial_params:
                        for param_name in self.initial_params[name]:
                            if f"{name}_{param_name}_change" in self.history and self.history[f"{name}_{param_name}_change"]:
                                print(f"{name} {param_name} change: {self.history[f'{name}_{param_name}_change'][-1]:.2f}%")
                                if isinstance(getattr(module, param_name, None), nn.Parameter):
                                    print(f"  Current {param_name}: {getattr(module, param_name).item():.4f}")
        
        # Store step
        if step is not None:
            self.steps.append(step)
        else:
            self.steps.append(len(self.steps))
            
        return changes

    def get_param_metrics(self):
        """Get metrics for learnable parameters (theta, rot_scale, rot_count)"""
        metrics = {}
        
        for name in self.initial_params.keys():
            for param_name in self.initial_params[name]:
                # Current value
                if f"{name}_{param_name}" in self.history and self.history[f"{name}_{param_name}"]:
                    metrics[f"{name}_{param_name}"] = self.history[f"{name}_{param_name}"][-1]
                
                # Change percentage
                if f"{name}_{param_name}_change" in self.history and self.history[f"{name}_{param_name}_change"]:
                    metrics[f"{name}_{param_name}_change"] = self.history[f"{name}_{param_name}_change"][-1]
        
        return metrics

    def get_freqs(self):
        """Return dictionary of current frequency values for all rotary embeddings"""
        freqs = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'freqs') and name in self.initial_freqs:
                freqs[name] = module.freqs.clone().detach().cpu()
        
        return freqs

    def get_current_freq_avg(self):
        """Return average frequency magnitude across all rotary embeddings for metrics display"""
        freqs = self.get_freqs()
        if not freqs:
            return 0.0
        
        # Calculate average magnitude across all modules
        total = 0.0
        count = 0
        for name, freq_tensor in freqs.items():
            if freq_tensor is not None:
                # Use absolute values to focus on magnitude
                avg = torch.abs(freq_tensor).mean().item()
                total += avg
                count += 1
        
        return total / max(count, 1)  # Avoid division by zero

    def plot_param_changes(self, save_path=None):
        """Plot changes in learnable parameters over time"""
        if not self.initial_params:
            print("No learnable parameters to plot")
            return
            
        param_names = set()
        for name in self.initial_params:
            param_names.update(self.initial_params[name].keys())
        
        # Create subplot for each parameter type
        fig, axes = plt.subplots(len(param_names), 1, figsize=(12, 4*len(param_names)))
        if len(param_names) == 1:  # Handle case of single subplot
            axes = [axes]
        
        for i, param_name in enumerate(param_names):
            ax = axes[i]
            ax.set_title(f"{param_name} Changes")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(f"{param_name} Value")
            
            for name in self.initial_params:
                if param_name in self.initial_params[name]:
                    if f"{name}_{param_name}" in self.history and len(self.history[f"{name}_{param_name}"]) > 0:
                        values = [v.item() if hasattr(v, 'item') else v for v in self.history[f"{name}_{param_name}"]]
                        ax.plot(self.steps, values, label=f"{name}")
            
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

  # these go in your model class

    # def create_frequency_tracker(self):
    #     from frequency_tracker import FrequencyTracker
    #     return FrequencyTracker(self)

    # def track_frequencies(self, step=None, verbose=False):
    #     if not hasattr(self, '_freq_tracker'):
    #         self._freq_tracker = self.create_frequency_tracker()
    #     return self._freq_tracker.check_changes(step, verbose)

    # def get_frequency_metrics(self):
    #     if hasattr(self, '_freq_tracker'):
    #         metrics = self._freq_tracker.get_metrics() if hasattr(self._freq_tracker, 'get_metrics') else {}
    #         param_metrics = self._freq_tracker.get_param_metrics() if hasattr(self._freq_tracker, 'get_param_metrics') else {}
    #         metrics.update(param_metrics)
    #         return metrics
    #     return {}
        
    def get_current_freq_value(self):
        if hasattr(self, '_freq_tracker'):
            return self._freq_tracker.get_current_freq_avg()
        return 0.0
