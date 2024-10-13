def start_train(cf, aisdatasets, aisdls):
  # Log in to your W&B account
  if cf.wdb:
    wandb.login()
    #Initialize wandb run
    wandb.init(project="team-tesi",
                name = 'test',
                id= 'test',
                resume=True)
    #Create a wandb Table to log images
    table_wdb_traj = wandb.Table(columns=["img_1", "img_2","img_3", "img_4","img_5", "img_6", "img_7"])
    table_wdb_sog = wandb.Table(columns=["img_1", "img_2","img_3", "img_4"])
  ## Model
  #===============================
  model = SeaFormer(cf, partition_model=None)

  ## Trainer
  #===============================
  epoch = 0
  if cf.val_dataset==True:
      #print("val_true")
      trainer = Trainer(cf, model, epoch, aisdatasets["train"], aisdatasets["validation"], savedir=cf.savedir, device=cf.device,aisdls=aisdls)
  else:
      #print("val_false")
      trainer = Trainer(cf, model, epoch, aisdatasets["train"], aisdatasets["test"], savedir=cf.savedir, device=cf.device,aisdls=aisdls)

  ## Training
  #================================
  if cf.retrain:
      trainer.train()
