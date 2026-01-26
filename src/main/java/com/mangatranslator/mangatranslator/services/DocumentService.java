package com.mangatranslator.mangatranslator.services;

import com.mangatranslator.mangatranslator.dtos.DocumentResponseDto;
import com.mangatranslator.mangatranslator.models.Document;
import com.mangatranslator.mangatranslator.models.Job;
import com.mangatranslator.mangatranslator.repositories.DocumentRepository;
import com.mangatranslator.mangatranslator.repositories.JobRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.UUID;

@Service
public class DocumentService {

    @Autowired
    private DocumentRepository documentRepository;

    @Autowired
    private JobRepository jobRepository;

    public DocumentResponseDto createDocumentAndJob(MultipartFile file, String sourceLang, String targetLang) throws IOException {
        String documentId = UUID.randomUUID().toString();
        String jobId = UUID.randomUUID().toString();
        String inputPath = "storage/input/" + documentId + "_" + file.getOriginalFilename();

        Files.createDirectories(Paths.get("storage/input"));

        file.transferTo(new File(inputPath));

        Document doc = new Document();
        doc.setId(documentId);
        doc.setOriginalFilename(file.getOriginalFilename());
        doc.setInputPath(inputPath);
        doc.setMimeType(file.getContentType());
        doc.setSizeBytes(file.getSize());
        doc.setCreatedAt(LocalDateTime.now());
        this.documentRepository.save(doc);

        Job job = new Job();
        job.setId(jobId);
        job.setDocumentId(documentId);
        job.setSourceLang(sourceLang);
        job.setTargetLang(targetLang);
        job.setStatus("QUEUED");
        job.setProgress(0);
        job.setCreatedAt(LocalDateTime.now());
        this.jobRepository.save(job);

        return new DocumentResponseDto(documentId, jobId, "QUEUED");
    }
}
