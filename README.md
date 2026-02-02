# VLA Manipulation for Unitree G1

**McMaster Humanoid Project**  
**Demo Date**: February 8, 2026

Using Visual-Language-Action (VLA) models with LeRobot + Genesis for G1 humanoid manipulation.

---

## 🎯 Quick Start

```bash
# Activate environment
cd ~/one-leg-robot
source vla_env/bin/activate

# Test G1 environment
python feb8_demo/test_g1_quick.py
```

---

## 📚 Documentation

**Start here**: `docs/README.md` - Complete documentation index

**Key files**:
- `docs/FEB8_SUCCESS_PLAN.md` - 14-day demo plan ⭐
- `docs/vla_setup_short.md` - Quick technical reference
- `docs/FEB8_DEMO_GENESIS.md` - Genesis implementation approach

---

## 📁 Project Structure

```
one-leg-robot/
├── docs/                    # All documentation
│   ├── README.md           # Documentation index
│   ├── FEB8_SUCCESS_PLAN.md # Master plan
│   └── ...
├── g1_manipulation/         # G1 environment (main code)
│   ├── env.py              # G1ManipulationEnv
│   └── test_env.py         # Tests
├── feb8_demo/               # Demo scripts
│   ├── test_g1_quick.py    # Quick test
│   └── test_g1_with_viewer.py # Visual test
├── rl-training-genesis/     # G1 model & training
│   └── models/
│       ├── g1_12dof.urdf   # Robot model
│       └── meshes/         # 64 STL files
└── vla_env/                 # Python environment
```

---

## ✅ Status (Feb 2, 2026)

```
✅ LeRobot 0.4.3 installed & working
✅ Genesis 0.3.13 simulation ready
✅ G1 meshes loaded (64 files)
✅ G1 loads successfully in Genesis
✅ ACT training proven (PushT: 94% loss reduction)
✅ G1ManipulationEnv created
🔄 Ready for demonstration recording
```

---

## 🚀 Next Steps

1. **Record demonstrations** (30-50 episodes)
2. **Train ACT policy** on demonstrations
3. **Evaluate policy** in simulation
4. **Prepare Feb 8 demo** presentation

See `docs/FEB8_SUCCESS_PLAN.md` for detailed timeline.

---

## 🛠️ Tech Stack

- **Physics**: Genesis 0.3.13 (GPU-accelerated)
- **ML Framework**: LeRobot 0.4.3 (HuggingFace)
- **Policy**: ACT (Action Chunking Transformer)
- **Robot**: Unitree G1 (12DOF version, 23 total DOF)
- **Hardware**: RTX 4070 GPU

---

## 📖 Key Achievements

- **PushT Training**: 94% loss reduction in 11 minutes
- **G1 Integration**: Full robot model with meshes in Genesis
- **Environment**: Gymnasium-compatible manipulation environment
- **Documentation**: Comprehensive setup and troubleshooting guides

---

## 🆘 Need Help?

1. **Setup issues**: See `docs/vla_setup_short.md`
2. **Planning questions**: See `docs/FEB8_SUCCESS_PLAN.md`
3. **Technical details**: See `docs/README.md` (documentation index)

---

**Project**: McMaster Humanoid Team  
**Last Updated**: February 2, 2026
