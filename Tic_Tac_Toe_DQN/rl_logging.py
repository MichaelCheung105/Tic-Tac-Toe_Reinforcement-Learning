from collections import defaultdict

logs = {'result': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))),
        'sampled_experience': defaultdict(lambda: {'cumulative_info': {'cumulative_seen_states': set()},
                                                   'per_sampling_info': defaultdict(lambda: defaultdict(lambda: None))
                                                   }
                                          ),
        'others': None
        }

