package com.mangatranslator.mangatranslator.controllers;

import com.mangatranslator.mangatranslator.dtos.DocumentResponseDto;
import com.mangatranslator.mangatranslator.services.DocumentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController("/api/document")
public class DocumentController {

    @Autowired
    private DocumentService documentService;

    @PostMapping("/create")
    public ResponseEntity<DocumentResponseDto> create(@RequestParam("file") MultipartFile file, @RequestParam("sourceLang") String sourceLang, @RequestParam("targetLang") String targetLang) throws IOException {
        return new ResponseEntity<DocumentResponseDto>(this.documentService.createDocumentAndJob(file,sourceLang,targetLang), HttpStatus.OK);
    }
}
