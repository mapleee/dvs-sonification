import pretty_midi
from config.load_yaml import configs
from midi2audio import FluidSynth

sf2 = FluidSynth(configs["sf2_path"])


def save_all_timbre():
    for i in range(128):
        c_chord = pretty_midi.PrettyMIDI()
        instr = pretty_midi.Instrument(program=i)
        for j, note_name in enumerate(['C5', 'E5', 'G5']):
            note_number = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=2, end=2.5)
            instr.notes.append(note)
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=j*0.5, end=(j+1)*.5)
            instr.notes.append(note)
        c_chord.instruments.append(instr)
        instr_name = pretty_midi.program_to_instrument_name(i)
        midi_path = f'data/timbre/midi/{i}_{instr_name}.mid'
        c_chord.write(midi_path)
        audio_path = f'data/timbre/audio/{i}_{instr_name}.wav'
        sf2.midi_to_audio(midi_path, audio_path)
