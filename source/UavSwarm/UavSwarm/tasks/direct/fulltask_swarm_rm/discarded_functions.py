def update_curriculum_stage(self):
        step = self.global_step
        c = self.cfg.curriculum
        
        if step <= c.stage1_end:
            if self.curriculum_stage != 1:
                print("Changed to curriculum stage 1: Individual Hover")
            self.curriculum_stage = 1
            self.cfg.episode_length_s = c.stage1_episode_length_s
            
        elif step <= c.stage2_end:
            if self.curriculum_stage != 2:
                print("Changed to curriculum stage 2: Individual Point-to-Point")
            self.curriculum_stage = 2
            self.cfg.episode_length_s = c.stage2_episode_length_s
            
        elif step <= c.stage3_end:
            if self.curriculum_stage != 3:
                print("Changed to curriculum stage 3: Individual Point-to-Point with Obstacles")
            self.curriculum_stage = 3
            self.cfg.episode_length_s = c.stage3_episode_length_s
            
        elif step <= c.stage4_end:
            if self.curriculum_stage != 4:
                print("Changed to curriculum stage 4: Swarm Navigation without Obstacles")
            self.curriculum_stage = 4
            self.cfg.episode_length_s = c.stage4_episode_length_s
        
        elif step <= c.stage5_end:
            if self.curriculum_stage != 5:
                    print("Changed to curriculum stage 5: Swarm Navigation with Obstacles")
            self.curriculum_stage = 5
            self.cfg.episode_length_s = c.stage5_episode_length_s  
        else:
            #print("Resetting curriculum to stage 1 after completion of all stages.")
            self.curriculum_stage = 1
            self.global_step = 0
            self.cfg.episode_length_s = c.stage1_episode_length_s


